


# Modern Computer Vision with PyTorch

<a href="https://www.packtpub.com/product/modern-computer-vision-with-pytorch/9781839213472?utm_source=github&utm_medium=repository&utm_campaign=9781839213472"><img src="https://static.packt-cdn.com/products/9781839213472/cover/smaller" alt="Modern Computer Vision with PyTorch" height="256px" align="right"></a>

This is the code repository for [Modern Computer Vision with PyTorch](https://www.packtpub.com/product/modern-computer-vision-with-pytorch/9781839213472?utm_source=github&utm_medium=repository&utm_campaign=9781839213472), published by Packt.

**Explore deep learning concepts and implement over 50 real-world image applications**

## What is this book about?
Deep learning is the driving force behind many recent advances in various computer vision (CV) applications. This book takes a hands-on approach to help you to solve over 50 CV problems using PyTorch1.x on real-world datasets.

By the end of this book, you’ll be able to leverage modern NN architectures to solve over 50 real-world computer vision problems confidently.

This book covers the following exciting features: 
* Train a NN from scratch in NumPy and then in PyTorch
* Implement 2D and 3D multi-object detection and segmentation
* Generate digits and DeepFakes with autoencoders and advanced (GANs)
* Manipulate images using CycleGAN, Pix2PixGAN, StyleGAN2, and SRGAN
* Combine CV with natural language processing to perform OCR, image captioning, and object detection
* Combine CV with reinforcement learning to build agents that play pong and self-drive a car
* Deploy a deep learning model on the AWS server using FastAPI and Docker
* Implement over 35 NN architectures and common OpenCV utilities

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1839213477) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" alt="https://www.packtpub.com/" border="5" /></a>
## Errata
- [Chapter01](Chapter01/readme.md)
- [Chapter03](Chapter03/Readme.md)
- [Chapter06](Chapter06/Readme.md)

## Instructions and Navigations
All of the code is organized into folders.

The code will look like the following:
```
def accuracy(x, y, model):
  model.eval() # <- let's wait till we get to dropout section
  # get the prediction matrix for a tensor of `x` images
  prediction = model(x)
  # compute if the location of maximum in each row coincides
  # with ground truth
  max_values, argmaxes = prediction.max(-1)
  is_correct = argmaxes == y
  return is_correct.cpu().numpy().tolist()
```

**Following is what you need for this book:**
This book is for beginners to PyTorch and intermediate-level machine learning practitioners who are looking to get well-versed with computer vision techniques using deep learning and PyTorch. If you are just getting started with neural networks, you’ll find the use cases accompanied by notebooks in GitHub present in this book useful. Basic knowledge of the Python programming language and machine learning is all you need to get started with this book. 

With the following software and hardware list you can run all code files present in the book (Chapter 1-18).

### Software and Hardware List

| Chapter  | Software required                                                                    | OS required                        |
| -------- | -------------------------------------------------------------------------------------| -----------------------------------|
| 1 - 18   |  Minimum 8 GB RAM, Intel i5 processor or better                                      | Windows, Mac OS X, and Linux (Any) |
|          |  NVIDIA 8+ GB graphics card – GTX1070 or better                                      |                                    |
|          |  Minimum 50 Mbps internet speed                                                      |                                    |
|          |   Python 3.6 and above                                                               |                                    |
|          |  PyTorch 1.7                                                                         |                                    |
|          |  Google Colab (can run in any browser)                             						      |                                    |

All the notebooks can be run directly on [colab](https://colab.google.com) using the [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/) button that can be found at the start of every notebook.

If you wish to run the notebooks locally, ensure you have a CUDA compatible GPU with drivers installed. Instructions are given [here](https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/blob/master/Install-CUDA-Drivers.md)

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781839213472_ColorImages.pdf).


### Related products <Other books you may enjoy>
* Hands-On Natural Language Processing with PyTorch 1.x [[Packt]](https://www.packtpub.com/product/hands-on-natural-language-processing-with-pytorch-1-x/9781789802740) [[Amazon]](https://www.amazon.com/dp/1789802741)

* PyTorch Computer Vision Cookbook [[Packt]](https://www.packtpub.com/product/pytorch-computer-vision-cookbook/9781838644833) [[Amazon]](https://www.amazon.com/dp/B0862CX2ZL)

## Get to Know the Author
**V Kishore Ayyadevara** leads a team focused on using AI to solve problems in the healthcare space. He has more than 10 years' experience in the field of data science with prominent technology companies. In his current role, he is responsible for developing a variety of cutting-edge analytical solutions that have an impact at scale while building strong technical teams. Kishore has filed 8 patents at the intersection of machine learning, healthcare, and operations. Prior to this book, he authored four books in the fields of machine learning and deep learning. Kishore got his MBA from IIM Calcutta and his engineering degree from Osmania University.

**Yeshwanth Reddy** is a senior data scientist with a strong focus on the research and implementation of cutting-edge technologies to solve problems in the health and computer vision domains. He has filed four patents in the field of OCR. He also has 2 years of teaching experience, where he delivered sessions to thousands of students in the fields of statistics, machine learning, AI, and natural language processing. He has completed his MTech and BTech at IIT Madras.


### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781839213472">https://packt.link/free-ebook/9781839213472 </a> </p>