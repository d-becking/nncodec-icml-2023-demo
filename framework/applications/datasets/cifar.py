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

from torchvision import datasets, transforms

class CIFAR100dataset(datasets.CIFAR100):
    def __init__(self, *args, validate=False, train=True, **kwargs):
        if train and validate == train:
            raise ValueError('Train and validate can not be True at the same time.')
        _train = train
        if validate:
            train = True
        super().__init__(*args, train=train, **kwargs)

        if _train:
            self.data = self.data[:40000]
        elif validate:
            self.data = self.data[40000:]

def cifar100_dataloaders(root, split='test', val_data_required=False):

    normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                     std=[0.2673, 0.2564, 0.2762])

    train_trafo = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize
            ])

    val_trafo = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

    if split == 'train':
        train_data = datasets.CIFAR100(root=root, train=True, transform=train_trafo, download=True)
        if val_data_required:
            train_data.data = train_data.data[10000:, :, :, :]
            train_data.targets = train_data.targets[10000:]
        return train_data

    elif split == 'val':
        val_data = datasets.CIFAR100(root=root, train=True, transform=val_trafo, download=True)
        val_data.data = val_data.data[:10000, :, :, :]
        val_data.targets = val_data.targets[:10000]
        return val_data

    elif split == 'test':
        test_data = datasets.CIFAR100(root=root, train=False, transform=val_trafo, download=True)
        return test_data