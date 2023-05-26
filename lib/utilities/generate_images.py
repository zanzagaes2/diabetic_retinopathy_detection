from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

class VisualizationGenerator(ABC):

    @abstractmethod
    def create_visualization(self, loader):
        return


class AttentionGenerator(VisualizationGenerator):

    def __init__(self, model, model_checkpoint):
        self.model = model
        self.model.load_state_dict(torch.load(model_checkpoint)["state_dict"])
        self.model = self.model.cuda().eval()

        self.attention = []
        hook = lambda _, __, output: self.attention.append(output.cpu())
        for b in self.model.blocks:
            b.attn.attn_drop.register_forward_hook(hook)

    def create_visualization(self, loader, path):
        self.attention = []
        names = []
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for batch, value, image_name in tqdm(loader):
                    names.append(image_name)
                    batch = batch.cuda()
                    self.model.forward_features(batch)

        for i, [name] in enumerate(names):
            att = self.attention[i * 12: (i + 1) * 12]
            image = Image.open(f"{path}/{name}").resize((224, 224))
            mask = AttentionGenerator.rollout(att, 0.8, "mean")
            np_image = np.array(image)[:, :, ::-1]
            mask = cv2.resize(mask, (np_image.shape[1], np_image.shape[0]))
            mask = AttentionGenerator.overlap_mask(np_image, mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            yield name, mask

    @staticmethod
    def rollout(attentions, discard_ratio, head_fusion):
        result = torch.eye(attentions[0].size(-1))
        with torch.no_grad():
            for attention in attentions:
                match head_fusion:
                    case "mean":
                        attention_heads_fused = attention.mean(axis=1)
                    case "max":
                        attention_heads_fused = attention.max(axis=1)[0]
                    case "min":
                        attention_heads_fused = attention.min(axis=1)[0]
                    case _:
                        raise "Attention head fusion type Not supported"

                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
                indices = indices[indices != 0]
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0 * I) / 2
                a = a / a.sum(dim=-1)

                result = torch.matmul(a, result)

        mask = result[0, 0, 1:]
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)
        return mask

    @staticmethod
    def overlap_mask(image, mask):
        image = np.float32(image) / 255
        alpha = np.where(mask > np.mean(mask), 0.25, 0)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = image + np.multiply(heatmap, alpha.reshape(*image.shape[:-1], 1))
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

class AttentionGradGenerator(VisualizationGenerator):

    def __init__(self, model, model_checkpoint):
        self.model = model
        self.model.load_state_dict(torch.load(model_checkpoint)["state_dict"])
        self.model = self.model.cuda().eval()

        self.attention = []
        self.attention_gradients = []
        get_attention = lambda _, __, output: self.attention.append(output.detach().cpu())
        get_attention_gradient = lambda _, grad_input, __: self.attention_gradients.append(grad_input[0].detach().cpu())
        for b in self.model.blocks:
            b.attn.attn_drop.register_forward_hook(get_attention)
            b.attn.attn_drop.register_full_backward_hook(get_attention_gradient)

    def create_visualization(self, loader, path):
        for param in self.model.parameters():
            param.requires_grad = True

        names = []
        self.attention = []
        for batch, value, image_name in tqdm(loader):
            names.append(image_name)
            batch = batch.cuda()
            value = value.cuda()

            output = self.model(batch)
            category_mask = torch.zeros(output.size()).cuda()
            category_mask[:, value] = 1
            loss = (output * category_mask).sum()
            loss.backward()
            self.model.zero_grad()

        for i, [name] in enumerate(names):
            att = self.attention[i * 12: (i + 1) * 12]
            grads = self.attention_gradients[i * 12: (i + 1) * 12]

            image = Image.open(f"{path}/{name}").resize((224, 224))
            mask = AttentionGradGenerator.grad_rollout(att, grads, 0.9)
            np_img = np.array(image)[:, :, ::-1]
            mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
            mask = AttentionGenerator.overlap_mask(np_img, mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            yield name, mask

    @staticmethod
    def grad_rollout(attention, gradients, discard_ratio):
        result = torch.eye(attention[0].size(-1))
        with torch.no_grad():
            for attention, grad in zip(attention, gradients):
                weights = grad
                attention_heads_fused = (attention * weights).mean(axis=1)
                attention_heads_fused[attention_heads_fused < 0] = 0

                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0 * I) / 2
                a = a / a.sum(dim=-1)
                result = torch.matmul(a, result)

        mask = result[0, 0, 1:]
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)
        return mask

class HeatmapGenerator(VisualizationGenerator):

    def __init__(self, model, model_checkpoint):
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        weights = torch.load(model_checkpoint)["state_dict"]
        self.model.load_state_dict(weights, strict=False)
        self.model = self.model.cuda().eval()

    def create_visualization(self, loader, path):
        convs = []
        names = []
        targets = []
        for batch, value, image_name in loader:
            batch = batch.to("cuda")
            names.extend(image_name)
            convs.append(self.model.forward_features(batch).cpu())
            targets.append(np.argmax(self.model(batch).cpu()))
        for name, conv, target in zip(names, convs, targets):
            image = Image.open(f"{path}/{name}").resize((256, 256))

            w = list(self.model.classifier.parameters())[0].cpu()
            conv = conv.squeeze()

            mask = HeatmapGenerator.heatmap(w, conv, target)
            np_img = np.array(image)[:, :, ::-1]
            mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
            mask = AttentionGenerator.overlap_mask(np_img, mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            yield name, mask

    @staticmethod
    def heatmap(weights, conv, target, discard_ratio = .8):
        weights = weights[target, :].numpy()
        conv = conv.numpy().transpose((1, 2, 0))
        heatmap = np.dot(conv, weights)
        heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap) - np.min(heatmap))

        flat = torch.tensor(heatmap.flatten())
        _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
        indices = indices[indices != 0]
        flat[indices] = 0

        return heatmap

def annotate(name, path, **kwargs):
    name = name
    with Image.open(f"{path}/{name}") as img:
        draw = ImageDraw.Draw(img)
        properties = "\n".join(f"{key} : {value}" for key, value in kwargs.items())
        label = f"Name: {name}\n" + properties
        draw.text((10, 5), label, (255,255,255))
    return img