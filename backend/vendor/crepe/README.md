# CREPE Vendor Folder

This folder stores the manually downloaded CREPE *full* capacity model.

1. Download [`model-full.h5`](https://github.com/marl/crepe/raw/master/assets/model-full.h5).
2. Place it at `backend/vendor/crepe/model-full.h5` (or copy to `models/crepe/model-full.h5`).
3. Restart the backend so that the loader picks up the weights.

The code only attempts to load the model if the file exists and otherwise
raises a structured `Model missing: crepe` error.
