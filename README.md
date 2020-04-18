# Detecting-COVID-19-in-X-ray-images-with-Keras-TensorFlow-and-Deep-Learning
how COVID-19 could be detected in chest X-rays of patients.  From there, we’ll review our COVID-19 chest X-ray dataset.  I’ll then show you how to train a deep learning model using Keras and TensorFlow to predict COVID-19 in our image dataset.

Given that there are limited COVID-19 testing kits, we need to rely on other diagnosis measures.

For the purposes of this tutorial, I thought to explore X-ray images as doctors frequently use X-rays and CT scans to diagnose pneumonia, lung inflammation, abscesses, and/or enlarged lymph nodes.

Since COVID-19 attacks the epithelial cells that line our respiratory tract, we can use X-rays to analyze the health of a patient’s lungs.

And given that nearly all hospitals have X-ray imaging machines, it could be possible to use X-rays to test for COVID-19 without the dedicated test kits.

A drawback is that X-ray analysis requires a radiology expert and takes significant time — which is precious when people are sick around the world. Therefore developing an automated analysis system is required to save medical professionals valuable time.

Note: There are newer publications that suggest CT scans are better for diagnosing COVID-19, but all we have to work with for this tutorial is an X-ray image dataset. Secondly, I am not a medical expert and I presume there are other, more reliable, methods that doctors and medical professionals will use to detect COVID-19 outside of the dedicated test kits.

Our COVID-19 patient X-ray image dataset
![Header Image](./covid19_keras_dataset.png)

My Project Glimpse
![Header Image](./upload.png)
Prediction Page
![Header Image](./prediction.png)


We simply don’t have enough (reliable) data to train a COVID-19 detector.

Hospitals are already overwhelmed with the number of COVID-19 cases, and given patients rights and confidentiality, it becomes even harder to assemble quality medical image datasets in a timely fashion.

I imagine in the next 12-18 months we’ll have more high quality COVID-19 image datasets; but for the time being, we can only make do with what we have.

I have done my best (given my current mental state and physical health) to put together a tutorial for my readers who are interested in applying computer vision and deep learning to the COVID-19 pandemic given my limited time and resources; however, I must remind you that I am not a trained medical expert.

For the COVID-19 detector to be deployed in the field, it would have to go through rigorous testing by trained medical professionals, working hand-in-hand with expert deep learning practitioners. The method covered here today is certainly not such a method, and is meant for educational purposes only.

Furthermore, we need to be concerned with what the model is actually “learning”.
