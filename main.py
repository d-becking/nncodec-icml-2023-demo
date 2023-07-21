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

import argparse
import nnc
import os
import shutil
import wandb
import torch
import numpy as np
import random

from framework.use_case_init import use_cases
from framework.pytorch_model import __initialize_data_functions, np_to_torch, torch_to_numpy
from framework.applications.utils.train import train_classification_model
from framework.applications.utils.evaluation import evaluate_classification_model, vis_feature_maps
from framework.applications import models, datasets
from ptflops import get_model_complexity_info
import torchvision

parser = argparse.ArgumentParser(description='NNCodec Experiments')
parser.add_argument('--qp', type=int, default=-36, help='quantization parameter (default: -36)')
parser.add_argument('--nonweight_qp', type=int, default=-75, help='qp for non-weights, e.g. BatchNorm params (default: -75)')
parser.add_argument("--opt_qp", action="store_true", help='Modifies QP layer-wise')
parser.add_argument("--use_dq", action="store_true", help='Enable dependent scalar / Trellis-coded quantization')
parser.add_argument("--lsa", action="store_true", help='Enable Local Scaling Adaptation')
parser.add_argument("--bnf", action="store_true", help='Enable BatchNorm Folding')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (default=64)')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train (default: 5)')
parser.add_argument('--max_batches', type=int, default=None, help='Max num of batches to process (default: 0, i.e., all)')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
parser.add_argument('--model', type=str, default='resnet56', metavar=f'any of {models.__all__} or {torchvision.models.list_models(torchvision.models)}')
parser.add_argument('--model_path', type=str, default=None, metavar='./example/ResNet56_CIF100.pt')
parser.add_argument('--dataset', type=str, default='CIFAR100dataset', metavar=f"Any of {datasets.__all__}")
parser.add_argument('--dataset_path', type=str, default='../data')
parser.add_argument('--results', type=str, default='./results')
parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers (default: 4)')
parser.add_argument("--wandb", action="store_true", help='Use Weights & Biases for data logging')
parser.add_argument('--wandb_key', type=str, default='',
                       help='Authentication key for Weights & Biases API account ')
parser.add_argument('--wandb_run_name', type=str, default='GPUcluster', help='Identifier for current run')
parser.add_argument("--pre_train_model", action="store_true", help='Training the full model prior to compression')
parser.add_argument("--plot_feature_maps", action="store_true", help='Plot features, i.e., in- and output activations')
parser.add_argument("--plot_segmentation_masks", action="store_true", help='Plot predicted segmentation masks')
parser.add_argument("--verbose", action="store_true", help='Stdout process information.')

def main():
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### model determinism / reproducibility
    torch.manual_seed(808)
    random.seed(909)
    np.random.seed(303)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if int(torch.version.cuda.split(".")[0]) > 10 or \
            (int(torch.version.cuda.split(".")[0]) == 10 and int(torch.version.cuda.split(".")[1]) >= 2):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if not os.path.exists(args.results):
        os.makedirs(args.results)

    if args.wandb:
        if isinstance(args.wandb_key, str) and len(args.wandb_key) == 40:
            os.environ["WANDB_API_KEY"] = args.wandb_key
        else:
            assert 0, "incompatible W&B authentication key"

    if args.model in models.__all__:
        model = models.init_model(args.model, num_classes=100)
    elif args.model in torchvision.models.list_models(torchvision.models):
        model = torchvision.models.get_model(args.model, weights="DEFAULT")
    elif args.model in torchvision.models.list_models(torchvision.models.segmentation):
        model = torchvision.models.get_model(args.model, weights="DEFAULT")
    else:
        assert 0, f"Model not specified in /framework/applications/models and not available in torchvision model zoo" \
                  f"{torchvision.models.list_models(torchvision.models)})"

    if args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))

    if args.wandb:
        wandb.init(
            config=args,
            project=f"{args.model}_{args.dataset}_{args.wandb_run_name}",
            name=f"qp_{args.qp}_opt_{args.opt_qp}_dq_{args.use_dq}_bnf_{args.bnf}_lsa_{args.lsa}",
            entity="edl-group",
            save_code=True,
            dir=f"{args.results}"
        )
        wandb.log({f"orig_{n}": v for n, v in model.state_dict().items() if not "num_batches_tracked" in n})

    criterion = torch.nn.CrossEntropyLoss()
    UCS = {'CIFAR100dataset': 'NNR_PYT_CIF100', 'VOC': 'NNR_PYT_VOC'}
    use_case_name = UCS[args.dataset] if args.dataset in UCS else 'NNR_PYT'

    test_set, test_loader, val_set, val_loader, train_loader = __initialize_data_functions(handler=use_cases[use_case_name],
                                                                                           dataset_path=args.dataset_path,
                                                                                           batch_size=args.batch_size,
                                                                                           num_workers=args.workers)
    for idx, (i, l) in enumerate(test_loader):
        if idx >= 1:
            break
        input_shape = tuple(i.shape[1:])
    macs, params = get_model_complexity_info(model, input_shape, as_strings=False,
                                             print_per_layer_stat=True, verbose=args.verbose,
                                             ignore_modules=[torch.nn.MultiheadAttention])

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    if args.plot_feature_maps:
        vis_feature_maps(model, test_loader, device=device, max_batches=args.max_batches, identifier="orig",
                         res_dir=args.results)

    test_perf, t5_miou, t_loss = evaluate_classification_model(model, criterion, test_loader, test_set, device=device,
                                                              verbose=args.verbose, max_batches=args.max_batches,
                                                              plot_segmentation_masks=args.plot_segmentation_masks)
    print(f"Initial test top1 acc: {test_perf}, {'mIoU' if args.dataset == 'VOC' else 'top5 acc'}: {t5_miou}, loss: {t_loss}")

    ### optionally pre-train the neural network model for args.epochs
    if args.pre_train_model:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.wandb:
            wandb.watch(model, log="all", log_graph=True)
        best_acc = 0
        for e in range(args.epochs):
            print(f"Epoch {e}")
            acc, loss, model = train_classification_model(model, optimizer, criterion, train_loader, device=device,
                                                          verbose=args.verbose, return_model=True)
            print(f"acc {acc}, mean_train_loss {loss}")
            test_perf, _, _ = evaluate_classification_model(model, criterion, test_loader, test_set, device=device, verbose=args.verbose)
            print(f"Test acc {test_perf}")
            if args.wandb:
                wandb.log({"acc": acc, "loss": loss, "test_loss": test_perf})
            if test_perf > best_acc:
                best_acc = test_perf
                torch.save(model.state_dict(), f"{args.results}/{model}_retrained.pt")

    ### compress the model
    bs, enc_mdl_info = nnc.compress_model(model,
                                          bitstream_path=f'{args.results}/{args.model}_qp_{args.qp}_bitstream.nnc',
                                          qp=args.qp,
                                          lsa=args.lsa,
                                          bnf=args.bnf,
                                          opt_qp=args.opt_qp,
                                          use_dq=args.use_dq,
                                          learning_rate=args.lr,
                                          epochs=args.epochs,
                                          use_case=use_case_name,
                                          dataset_path=args.dataset_path,
                                          wandb_logging=args.wandb,
                                          return_bitstream=True,
                                          return_model_data=True or args.bnf,
                                          max_batches=args.max_batches,
                                          num_workers=args.workers,
                                          batch_size=args.batch_size,
                                          verbose=args.verbose)

    ### decompress the bitstream
    rec_mdl_params = nnc.decompress(bs, verbose=args.verbose)

    ### reconstruction
    if args.bnf:
        for n, m in model.named_modules(): # reset runnning statistics and trainable bn_gamma to 1
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()
                m.weight = torch.nn.Parameter(torch.ones_like(m.weight))
        # re-name bn_beta-type params to PYT bn.bias
        rec_mdl_params = {enc_mdl_info["bnf_matching"][param] if param in enc_mdl_info["bnf_matching"]
                          else param: rec_mdl_params[param] for param in rec_mdl_params}

    ### evaluation of decoded and reconstructed model
    model.load_state_dict(np_to_torch(rec_mdl_params), strict=False if args.bnf else True)

    torch.save(model.state_dict(), f"{args.results}/{args.model}_dict_dec_rec.pt")

    if args.plot_feature_maps:
        vis_feature_maps(model, test_loader, device=device, max_batches=args.max_batches, identifier="rec", res_dir=args.results)

    test_perf, t5_miou, t_loss = evaluate_classification_model(model, criterion, test_loader, test_set, device=device,
                                                    verbose=args.verbose, max_batches=args.max_batches,
                                                    plot_segmentation_masks=args.plot_segmentation_masks,
                                                    prefix=f"_compressed_qp_{args.qp}",
                                                    orig_iou=test_perf)

    print(f"Initial test top1 acc: {test_perf}, {'mIoU' if args.dataset == 'VOC' else 'top5 acc'}: {t5_miou}, loss: {t_loss}")

    if args.wandb:
        wandb.log({"rec_mdl_params": model.state_dict(), "test_acc_rec": test_perf})
        wandb.finish()
        print(f"zipping W&B files to {args.results}/wandb_zipped")
        shutil.make_archive(f"{args.results}/wandb_zipped", 'zip', f"{args.results}/wandb")
        print("remove W&B dir")
        shutil.rmtree(f"{args.results}/wandb")

if __name__ == '__main__':
    main()
