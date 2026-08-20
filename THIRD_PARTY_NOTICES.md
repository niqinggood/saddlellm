# Third-party world-model notices

SaddleLLM's native `categorical_rssm` backend is an independently written
PyTorch implementation. Its categorical-state, probability-mixing, split-KL,
signed-log/two-hot, and grouped-recurrence design was informed by published
world-model techniques and study of the bundled DreamerV3 reference. No
reference project is imported, launched, or used as a runtime backend by the
SaddleLLM world-model framework.

## DreamerV3

Source: `reference/dreamerv3-main`  
License: MIT

Copyright (c) 2024 Danijar Hafner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Other bundled research references

- IRIS (`reference/iris-main`): GPL-3.0. See its `LICENSE`.
- Drive-OccWorld (`reference/Drive-OccWorld-main`): Apache-2.0. See its
  `LICENSE`; its MMDetection3D dependencies retain their own terms.
- HY-WorldPlay (`reference/HY-WorldPlay-main`): Tencent HY-WorldPlay Community
  License. See `License.txt`. It excludes the EU, United Kingdom, and South
  Korea and includes additional commercial, use, and model-output terms.
- WorldCompass (`reference/HY-WorldPlay-main/worldcompass`): its code is
  Apache-2.0, while the required HY-WorldPlay model/code remains subject to the
  Tencent HY-WorldPlay Community License.

These trees remain research material outside the native implementation. This
notice is informational and is not legal advice. Review the exact license files
before executing, modifying, or redistributing a reference project.
