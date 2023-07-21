'''
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or 
other Intellectual Property Rights other than the copyrights concerning 
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2019-2023, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

import torch
from torchvision.utils import make_grid, save_image, draw_segmentation_masks
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tqdm import tqdm
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from framework.applications.utils.metrics import get_topk_accuracy_per_batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_classification_model(model, criterion, testloader, testset,  min_sample_size=1000, max_batches=None,
                                  early_stopping_threshold=None, device=DEVICE, print_classification_report=False,
                                  return_predictions=False, verbose=False, plot_segmentation_masks=False, prefix='',
                                  orig_iou=None):
    """
    Helper function to evaluate model on test dataset.

    Parameters
    ----------
    model: torch.nn.Module
        Neural network model.
    criterion: torch.nn.Criterion
        Criterion for loss calculation.
    testloader: torch.utils.data.DataLoader
        DataLoader that loaded testset.
    testset: torch.utils.data.dataset.Dataset
        Test dataset
    min_sample_size: int
        Minimum sample size used for early_stopping calculation. Default: 1000
    max_batches: int
        Maximum batches evaluated, by default evaluates the complete testset. Default: None
    early_stopping_threshold: int
        A value between 0-100 corresponding to the accuracy. If it drops under a given threshold
        the evaluation is stopped.
    device: str
        Device on which the model is evaluated: cpu or cuda.
    print_classification_report: bool
        If True print the complete confusion matrix for all the classes.
    return_predictions: bool
        If True return all the predictions for all samples, otherwise return the accuracy.
    verbose: bool
        If True print the progress bar of the evaluation.

    Return
    ------
    output: float | nd.array
        Accuracy or all predictions, depending on the given return_predictions parameter.
    """
    model = model.to(device)
    model.eval()
    test_loss = []
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    top5_acc = 0

    # set (verbose) iterator
    total_iterations = max_batches or len(testloader)
    # iterator = tqdm(enumerate(testloader), total=total_iterations, position=0, leave=True) if verbose else enumerate(testloader)
    iterator = enumerate(testloader)

    DeepLab_condition = model.__class__.__name__ == "DeepLabV3" if not (isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel)) \
                            else model.module.__class__.__name__ == "DeepLabV3"

    if DeepLab_condition:
        from torchmetrics import JaccardIndex
        num_of_classes = model.classifier[len(model.classifier) - 1].weight.shape[0]
        jaccard = JaccardIndex(task="multiclass", num_classes=num_of_classes, average="macro").to(device)

    def show(imgs, mode="", miou=None, uncompressed_iou=None):
        if not isinstance(imgs, list):
            imgs = [imgs]
        factor = 3
        fig, axs = plt.subplots(nrows=1, ncols=len(imgs), squeeze=False, figsize=(len(imgs) * factor, 1.5 * factor), dpi=150)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            if miou is not None:
                if uncompressed_iou is not None:
                    iou_diff = miou[i] - uncompressed_iou[i]
                    iou_to_print = "\u0394" + f" = {iou_diff:.2f}%" if \
                            iou_diff < 0 else "\u0394" + f" = +{iou_diff:.2f}%"
                else:
                    iou_to_print = f"mIoU: {miou[i]:.2f}%"
                axs[0, i].text(imgs[0].shape[1] / 2, 10, iou_to_print, color='white', fontsize=20, ha='center', va='top')
            elif i == 0:
                axs[0, i].text(imgs[0].shape[1] / 2, 10, "Ground truth", color='white', fontsize=20, ha='center', va='top')

        plt.tight_layout(pad=0)
        plt.savefig(f"./segmask{prefix}_{mode}.png")
        plt.show()
        plt.close()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if DeepLab_condition:
                outputs = outputs['out']
                targets = targets * (targets != 1) * 255

                if testloader.batch_size == 1 and plot_segmentation_masks:
                    all_classes_masks = torch.max(outputs[0], dim=0)[1].cpu() == torch.arange(outputs.shape[1])[:, None, None]
                    input_image = torch.tensor(np.transpose(np.asarray(F.to_pil_image(inputs[0].cpu())), (2, 0, 1)))
                    segmask = draw_segmentation_masks(input_image, all_classes_masks, alpha=0.5)

                    if not "compressed" in prefix:
                        all_classes_target = targets.squeeze(1).cpu() == torch.arange(outputs.shape[1])[:, None, None]
                        segmask_gt = draw_segmentation_masks(input_image, all_classes_target, alpha=0.5)

                targets = targets.squeeze(1).long()

            loss = criterion(outputs, targets)

            if outputs.size(1) > 5 and not DeepLab_condition:
                c1, c5 = get_topk_accuracy_per_batch(outputs, targets, topk=(1, 5))
                top5_acc += c5 * targets.size(0)

            test_loss.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_predictions.append(np.array(predicted.cpu()))
            all_labels.append(np.array(targets.cpu()))

            acc = 100. * correct / total

            if DeepLab_condition:
                jaccard.update(predicted, targets)
                if verbose:
                    print('Running Test/Val mIOU (batch {}/{}) over all {} classes: {}'.format(batch_idx,
                                                                                               total_iterations,
                                                                                               num_of_classes,
                                                                                               jaccard.compute() * 100))
                if testloader.batch_size == 1 and plot_segmentation_masks:
                    classes_present = torch.unique(torch.concatenate((torch.unique(predicted),  torch.unique(targets))))
                    single_image_predicted = torch.zeros_like(predicted)
                    single_image_targets = torch.zeros_like(targets)
                    for i, dim in enumerate(classes_present):
                        single_image_predicted[predicted == dim] = i
                        single_image_targets[targets == dim] = i
                    jaccard_single = JaccardIndex(task="multiclass", num_classes=classes_present.shape[0], average="macro").to(device)
                    jaccard_single.update(single_image_predicted, single_image_targets)
                    single_img_iou = jaccard_single.compute() * 100
                    if verbose:
                        print('Single Image {}/{} Test/Val mIOU): {}'.format(batch_idx, total_iterations, single_img_iou))

                    if batch_idx == 0:
                        if not "compressed" in prefix:
                            segmask_gt_stack = [segmask_gt]
                        segmask_stack = [segmask]
                        single_img_iou_stack = [single_img_iou.item()]
                    elif batch_idx < len(testloader) - 1:
                        if not "compressed" in prefix:
                            segmask_gt_stack.append(segmask_gt)
                        segmask_stack.append(segmask)
                        single_img_iou_stack.append(single_img_iou.item())
                    else:
                        if not "compressed" in prefix:
                            show(segmask_gt_stack, mode="gt")
                        show(segmask_stack, miou=single_img_iou_stack, uncompressed_iou=orig_iou)

            if batch_idx == max_batches:
                break
            elif len(all_predictions) > min_sample_size and early_stopping_threshold is not None and \
                    acc < early_stopping_threshold:
                break

        acc = 100. * correct / total
        if top5_acc != 0:
            top5_acc = top5_acc / total

        if print_classification_report:
            print(classification_report(np.concatenate(all_labels), np.concatenate(all_predictions),
                                        target_names=list(testset.mapping.keys()),
                                        labels=list(testset.mapping.values())))

        if return_predictions:
            return np.concatenate(all_predictions)
        else:
            mean_test_loss = np.mean(test_loss)
            if DeepLab_condition:
                return single_img_iou_stack, jaccard.compute() * 100, mean_test_loss
            else:
                return acc, float(top5_acc), mean_test_loss

def evaluate_classification_model_TEF(model, test_loader, test_set, num_workers=8, verbose=0):

    _ , val_labels = zip(*test_set.imgs)

    y_pred = model.predict(test_loader, verbose=verbose, callbacks=None, max_queue_size=10, workers=num_workers,
                           use_multiprocessing=True)

    top5 = tf.keras.metrics.sparse_top_k_categorical_accuracy(val_labels, y_pred, k=5)
    top1 = tf.keras.metrics.sparse_categorical_accuracy(val_labels, y_pred)
    loss = tf.keras.metrics.sparse_categorical_crossentropy(val_labels, y_pred)

    acc = []
    acc.append((tf.keras.backend.sum(top1) / len(top1)).numpy() * 100)
    acc.append((tf.keras.backend.sum(top5) / len(top5)).numpy() * 100)
    acc.append((tf.keras.backend.mean(loss)).numpy())

    return acc

class FeatureMaps:
    def __init__(self, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_handler = OrderedDict()
        self.feature_handler_keys = []
        self.hook_module_id = 0
        for mod_name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):# or isinstance(module, torch.nn.Linear):
                module.register_forward_hook(self.hook_fct)
                self.feature_handler_keys.append(mod_name)
        self.num_hook_modules = len(self.feature_handler_keys)
    def hook_fct(self, param, i, o):
        self.hook_module_id += 1
        self.feature_handler[self.feature_handler_keys[self.hook_module_id - 1]] = {"in": i[0].data, "out": o.data}
        if self.hook_module_id == self.num_hook_modules:
            self.hook_module_id = 0
    def log_f_map(self, identifier="", sample_wise=True, root="./results"):
        path = f"{root}/exported_feature_maps/"
        if not os.path.exists(f"{root}/exported_feature_maps/{identifier}"):
            if not os.path.exists(path):
                os.mkdir(path)
            path += identifier
            if not os.path.exists(path):
                os.mkdir(path)
        else:
            path = f"{root}/exported_feature_maps/{identifier}"
        for l_idx, layer in enumerate(self.feature_handler_keys):
            if len(self.feature_handler[layer]["in"].shape) == 4: ## i.e., conv layers
                if sample_wise:
                    for sid, sample in enumerate(self.feature_handler[layer]["in"]):
                        dim = 0 if l_idx == 0 and sample.shape[0] == 3 else 1 ## input RGB image
                        save_image(make_grid(sample.unsqueeze(dim=dim), normalize=False), f"./{path}/{layer}_in_{sid}.png")
                        save_image(make_grid(sample.unsqueeze(dim=dim), normalize=True), f"./{path}/{layer}_in_{sid}_normalized.png")
                    for sid, sample in enumerate(self.feature_handler[layer]["out"]):
                        save_image(make_grid(sample.unsqueeze(dim=1), normalize=False), f"./{path}/{layer}_out_{sid}.png")
                        save_image(make_grid(sample.unsqueeze(dim=1), normalize=True), f"./{path}/{layer}_out_{sid}_normalized.png")
                else:
                    for channel in range(self.feature_handler[layer]["in"].shape[1]):
                        save_image(make_grid(self.feature_handler[layer]["in"], normalize=True)[channel],
                                   f"./{path}/{layer}_in_{channel}.png")
                    for o_channel in range(self.feature_handler[layer]["out"].shape[1]):
                        save_image(make_grid(self.feature_handler[layer]["out"], normalize=True)[o_channel],
                                   f"./{path}/{layer}_out_{o_channel}.png")
        self.feature_handler = OrderedDict()
def vis_feature_maps(model, testloader, device=DEVICE, max_batches=1, identifier="", res_dir="./results"):
    model = model.to(device)
    model.eval()
    iterator = enumerate(testloader)
    feature_maps = FeatureMaps(model)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            _ = model(inputs)
            feature_maps.log_f_map(identifier, root=res_dir)
            if batch_idx == max_batches:
                break

