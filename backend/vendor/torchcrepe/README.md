# TorchCREPE Vendor Folder

Offline TorchCREPE requires the `full.pth` weights file. Download it from
https://github.com/maxrmorrison/torchcrepe/raw/main/torchcrepe/assets/full.pth
and place it at `backend/vendor/torchcrepe/full.pth`.

The backend loads the weights only if the file is present and surfaces a
`Model missing: torchcrepe` error when absent.
