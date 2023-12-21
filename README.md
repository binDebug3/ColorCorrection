# ColorCorrection

**Readme**

Welcome to our project repository! In this codebase, we've implemented a state-of-the-art image generation system leveraging Generative Adversarial Networks (GANs) trained to add color to black and white images by Dallin Stewart and Gwen Martin. Our goal was to create a model capable of generating realistic and high-quality colored images by learning from a dataset of black and white landscape images.

**Overview:**
The core of our implementation revolves around GANs, a powerful deep learning technique that involves training two neural networks, a generator, and a discriminator, in a competitive manner. The generator aims to create realistic data, while the discriminator's role is to distinguish between real and generated data. Through this adversarial process, the generator continually improves its ability to produce convincing outputs.

**Dataset:**
We utilized a diverse and extensive dataset, carefully curated to cover a wide range of landscape styles and features. This dataset is crucial for training our GAN to generalize well and generate diverse and realistic images. It comes from [Landscape color and grayscale images](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization) by Kaggle user [_Bhabuk_](https://www.kaggle.com/theblackmamba31).

**Architecture:**
Our GAN architecture consists of a deep neural network for both the generator and discriminator. The generator synthesizes images, and the discriminator evaluates their realism. We experimented with various architectures, fine-tuning layers and parameters to achieve a balance between image quality and training stability. The architecture was inspired by a tutorial by [Moein Shariatnia](https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8). 

**Training Process:**
The training process involves an adversarial interplay between the generator and discriminator. As training progresses, the generator becomes adept at creating images that are increasingly indistinguishable from real ones. We carefully monitored training dynamics, preventing issues like mode collapse and ensuring a stable convergence. The output images are a result of training for 20 epochs, which took about 48 hours on BYU's supercomputer.

**Hyperparameter Tuning:**
Achieving optimal performance required thorough hyperparameter tuning. We experimented with learning rates, batch sizes, and regularization techniques to strike the right balance between fast convergence and avoiding overfitting.

**Results:**
Our GAN has demonstrated impressive results in generating high-quality images that closely resemble those in the training dataset. The model has learned intricate patterns, textures, and styles, making it a versatile tool for various creative applications. You can view an example below or in the output_images folder. The top row shows the original grayscale images, the bottom row displays the true or correct images, and the middle row contains the output images from our model.

**Future Work:**
As we continue to develop this project, future iterations may include enhancements such as progressive GANs, attention mechanisms, and novel loss functions to further improve image quality and diversity.

Thank you for exploring our image generation project! We hope our implementation inspires further exploration and development in the exciting field of generative adversarial networks.