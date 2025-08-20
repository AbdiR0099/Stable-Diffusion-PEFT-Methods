# Abstract
This paper addresses the challenge of adapting large text-to-image diffusion models for niche artistic style transfer, a task where a direct comparison of leading Parameter-Efficient Fine-Tuning (PEFT) methods is under-explored. The central issue investigated is the comparative efficacy of Low-Rank Adaptation (LoRA) and Textual Inversion (TI) for this purpose. The paper is structured to first detail the implementation of both PEFT methods, followed by a comparative evaluation. The methodology employs both quantitative analysis; assessing training efficiency, loss dynamics, CLIP, and FID scores and a qualitative user study evaluating aesthetic appeal and content preservation. The results demonstrate that LoRA achieves superior performance across all metrics. It exhibited more stable training dynamics, avoiding the overfitting observed in the Textual Inversion process, and attained better quantitative scores for image fidelity and prompt alignment. Furthermore, a blind user study revealed an overwhelming preference (90%) for LoRA-generated images. The paper concludes that LoRA is the more effective, precise, and robust methodology for high-fidelity style transfer under data-constrained conditions.

## Research Question
This leads to the focal research question that is, how do LoRA and TI compare in their efficacy, efficiency and precision in reproducing a niche image style using a pre-trained SD model that has been trained on a limited dataset? 

## Data Curation
115 landmark images from the Google Landmarks Dataset V2 were selected to ensure global diversity in architecture, scale and composition. This dataset is designed for fine-grained visual localization and contains real-world photographic variations such as landscape images or aerial view images, making it well suited as the base for stylization.
The 115 source landmark images were processed using the Stable Diffusion WEBUI, a popular open-source interface for the stable diffusion model. Using the img2img functionality, each image was re-rendered into a “Ghibli” style image using a text prompt that guided the model.

A critical component of the synthetic data generation phase was careful crafting of text prompts. For each image, a unique prompt was created and saved to a text file. The prompts followed a structured template:

                 ```<name of landmark> <location> <description> <description of composition>```

Image transformation for the suitable dataset is a crucial step. There are 3 transformations applied to each of the source image:
•	Resolution Standardization: This step resizes each image to 512*512 pixels. This is essential because neural networks require fixed-size inputs.
•	Centre Cropping: This function performs a crop to fit the target aspect ratio before resizing. This ensures that the most salient, central part of the landmark is preserved while potentially distracting or irrelevant border elements are removed.
•	High-Quality Resampling: The Lanczos filter is widely regarded as providing a superior trade-off between sharpness and reduction of aliasing artifacts making it an excellent choice for downscaling photographic images while preserving details.

## Low-Rank Adaptation (LoRA)
LoRA is a popular PEFT technique which is designed to efficiently adapt to large machine learning models to newer contexts. Its mechanism operates by isolating the original pre-trained model weights and parameters and by introducing trainable rank decomposition matrices, typically referred to as A and B, into selected layers of the model. The full weight update that occurs during a traditional fine-tuning stage is approximates based on the product of these two low-rank matrices.
This PEFT method significantly reduced the number of trainable parameters, for instance, it can reduce the trainable parameters for a GPT-3 model from 175 billion to approximately 18 million which leads to reduced utilization of GPU resources by estimated 66%. This in turns makes the training process efficient and fast while also lowering the hardware bottleneck for fine-tuning.

## Guide
### Step 1: Prepare the Dataset
This project uses a specific dataset format. The script will handle the preprocessing for you.

1. Input Format: Create a folder containing your images. For each image (e.g., `image1.png`), there must be a corresponding text file with the same name (e.g., `image1.txt`) containing the caption.

2. Preprocessing: The script automatically preprocesses the data by:

- Resizing each image to a resolution of 512x512 pixels.

- Applying a center crop to ensure consistent dimensions.

- Using the high-quality Lanczos resampling filter.

Output: The processed images are saved to a new folder. The script also splits your captions into `train.txt` and `validation.txt` files, which are essential for the training and validation phases.

### Step 2: Guide to LoRA Training
The core of this project is the LoRA training script. Here’s a breakdown of how it works.

1. Import Libraries: First, all the necessary libraries like PyTorch, Diffusers, and Transformers are imported.

2. Set Configuration: All training parameters are managed within the `TrainingConfig` class. An instance of this class is created to set up the entire training run. Key parameters include:

- `num_epochs:` The number of times to train on the entire dataset.

- `model_name:` The base Stable Diffusion model to fine-tune (e.g., `stable-diffusion-v1-5/stable-diffusion-v1-5`).

- `image_folder, train_file, val_file:` Paths to your preprocessed images and caption files.

- `batch_size` and `learning_rate:` Standard deep learning hyperparameters.

- `lora_rank` and `lora_alpha:` Key parameters that control the complexity and strength of the LoRA adaptation.

3. Create a Custom Dataset: The `GhibliVisDataset` class is used to load your images and captions efficiently. For each item, it returns a dictionary containing two tensors:

- `"pixel_values":` The processed image, ready for the model.

- `"input_ids":` The caption, tokenized (converted to numbers) for the text encoder.

4. Load the Stable Diffusion Architecture: The script loads the main components of the Stable Diffusion model:

- Tokenizer & Text Encoder: To process text prompts.

- Variational Autoencoder (VAE): To encode images into a compressed latent space.

- UNet: The core model that learns to denoise images.

- Noise Scheduler: Manages the addition of noise during training.

Crucially, the original weights of the VAE, Text Encoder, and UNet are frozen. Only the new, lightweight LoRA layers that are "injected" into the UNet will have their weights updated during training.

5. Prepare for Training: An optimizer (like AdamW) and a learning rate scheduler are created. The Hugging Face `accelerate` library is used to prepare all components for efficient training on the available hardware (CPU, GPU, etc.).

6. The Training Loop: The model learns over several epochs. In each step of an epoch, the model performs the following actions for a batch of data:

- Encode Text & Image: The text caption is converted into an embedding, and the image is compressed into a latent representation.

- Add Noise: A random amount of noise is added to the image latent, creating a "noisy" version.

- Predict the Noise: The UNet, guided by the text embedding, analyzes the noisy latent and predicts the noise that was originally added. This is the primary learning task.

- Calculate Loss: The model's prediction is compared against the actual noise. The difference between them (the loss) quantifies how well the model performed.

- Backpropagate & Optimize: The loss is used to adjust the weights of the LoRA layers only. This step teaches the model to become better at predicting the noise, guided by the text prompt.

- At the end of each epoch, this process is repeated on the validation set to check performance on unseen data without updating the model weights.

7. Save the LoRA Weights: After training is complete, the script extracts the small, trained LoRA weight layers from the UNet and saves them as a .safetensors file in a directory you provide. This file contains all the learned knowledge.

### Step 3: Generating Images (Inference)

Once you have your trained `.safetensors` file, you can use it to generate new images.

1. Load the Pipeline: Use the `AutoPipelineForText2Image` class from Diffusers to load the original base Stable Diffusion model.

2. Load LoRA Weights: Load your trained `.safetensors` file into the pipeline. This applies your fine-tuned adjustments on top of the base model.

3. Generate an Image: Provide a text prompt. The model will now generate an image that is influenced by the concepts and styles it learned during your fine-tuning process. The final image is saved to a directory you specify.

## ------ ##

The training process for the LoRA model was configured to run for 10 epochs. The training log provides a clear and consistent measure of its performance:

•	Average Time per Epoch: The duration for each epoch remained remarkably stable, averaging approximately 15 minutes and 40 seconds.

•	Average Time per Iteration: The time taken to process a single batch (step) averaged ~9.3 seconds/iteration (s/it).

•	Total Training Time: Based on the 10-epoch run, the total computational time for the LoRA fine-tuning process was approximately 2 hours and 40 minutes.

The final loss values for each epoch are represented below:

![LoRA Loss!](./LoRA-Training-Validation-Loss.png)

•	Initial Convergence: The model demonstrates a significant learning event between Epoch 1 and Epoch 2, where the validation loss drops dramatically from 0.2019 to 0.0907. This indicates that the model quickly began to understand the general patterns of the "ghiblivis" style.

•	Loss Fluctuation: Both the training and validation loss exhibit considerable fluctuation throughout the training process. This is expected and normal when fine-tuning on a small, diverse dataset. The model is continuously adjusting its weights based on different batches of images, which can cause temporary increases in loss as it encounters new or challenging examples.

•	Generalization Performance: A key observation is that from Epoch 2 onwards, the validation loss is frequently lower than the training loss. While this may seem counter-intuitive, it is often a sign of a well-regularized model. The LoRA configuration includes a dropout layer (lora_dropout=0.1), which is active during the training phase but disabled during validation. Dropout randomly deactivates a portion of the neurons, making the training task harder for the model. When validation is performed, the full, unobstructed model is used, which can result in a better (lower) loss score.

•	Final Convergence: The most critical result is the final epoch. The validation loss in Epoch 10 reaches 0.0596, the lowest point in the entire training run. This indicates that the model was still improving its generalization capabilities up to the final epoch and did not show signs of overfitting within this 10-epoch timeframe.

## Textual Inversion (TI)
This PEFT method is used to introduce new, user-defined concepts into text-to-image models like stable diffusion. This mechanism involves learning new “mock-words” or “contextual words” that are integrated as embeddings into the model’s textual embedding space. These new embeddings are optimized to characterize a specific detail or a visual concept within the image, such that they can be used within text prompts.

## Guide
### 1. Setup & Configuration
Dataset Format: Your caption files must be formatted with each line as `image_name.png|the original caption`. The script will automatically convert this into a full training prompt, like `"The image in <ghiblivis-style> is the original caption"`.

Configuration: Set your parameters in the `TrainingConfig class`. Instead of LoRA parameters, you will define:

- `placeholder_token`: The new "word" you will teach the model (e.g., `"<ghiblivis-style>"`). This token will represent your unique concept.

- `initializer_token`: An existing word (e.g., `"style"`) used as a starting point for your new token, which helps the model learn faster.

### 2. Model Preparation & Latent Pre-computation
1. Inject New Token: The script adds your `placeholder_token` to the model's vocabulary and initializes its embedding.

2. Freeze Model: For Textual Inversion, the entire model (VAE, UNet, and Text Encoder) is frozen, except for the single new token embedding. This ensures that only the meaning of your new word is learned, without changing the base model.

3. Pre-compute Latents: To drastically speed up training, the script runs a one-time process to convert all images in your dataset into latents.

- What are latents? Latents are a compressed, numerical "essence" of an image created by the VAE. They are much smaller and faster to process than full pixel images.

- Why is this efficient? By pre-computing and saving the latents to disk, the training loop can skip the repetitive and slow image-to-latent conversion in every step, making the training significantly faster.

- Can this be used for LoRA? No. Pre-computing latents is not suitable for LoRA because LoRA modifies the UNet, which needs to see variations in the latents created by data augmentations (like random cropping) applied to the full images. Since Textual Inversion only modifies the text embedding, the image latents can remain static.

### 3. Training
- Optimized Loop: The training loop loads batches of the pre-computed latents and their corresponding captions.

- The Learning Process: For each step, the model:

1. Adds noise to the pre-computed latents.

2. Uses the frozen UNet to predict that noise, guided by the text caption which includes your trainable `placeholder_token`.

3. Calculates the difference (loss) between the predicted and actual noise.

4. Uses this loss to update only the embedding for your new `placeholder_token`. All other weights in the model remain unchanged.

### 4. Saving and Inference
1. Save the Embedding: After training, the script extracts the single, newly learned embedding vector for your `placeholder_token` and saves it as a very small `.safetensors` file.

2. Generate Images:

Load a standard Stable Diffusion pipeline (e.g., `runwayml/stable-diffusion-v1-5`).

Load your custom concept using `pipe.load_textual_inversion("path/to/your.safetensors")`.

Write a prompt that includes your `placeholder_token` (e.g., "A photo of Peterborough Cathedral in `<ghiblivis-style>`") to generate an image that incorporates your learned concept.

## ------ ##

The Textual Inversion experiment was configured to run for 15 epochs and incorporated a significant optimization: latent pre-computation. The training log is therefore segmented into a training phase and a validation phase for each epoch.

•	Training Phase Analysis:
Average Time per Epoch (Training): The training phase for each epoch was significantly faster than LoRA's, averaging approximately 12 minutes and 20 seconds. 
Average Time per Iteration (Training): The per-iteration latency was also substantially lower, averaging ~7.3 seconds/iteration.

•	Validation Phase Analysis:
Average Time per Epoch (Validation): The validation phase was extremely fast, consistently taking approximately 38 seconds per epoch.
Average Time per Iteration (Validation): The per-iteration latency during validation averaged ~3.5 seconds/iteration.

•	Total TI Performance:
Total Time per Epoch: By summing the training and validation phases, the total time per epoch for Textual Inversion was approximately 12 minutes and 58 seconds.
Total Training Time: Over its 15-epoch run, the total computational time for the Textual Inversion fine-tuning process was approximately 3 hours and 15 minutes.

The Textual Inversion model was trained for 15 epochs. The final loss values for each epoch are represented below:
 
![TI Loss!](./TI-Training-Validation-Loss.png)

•	Training Loss Trend: The training loss for Textual Inversion shows a more consistent, albeit noisy, downward trend over the 15 epochs, starting at 0.1469 and ending at 0.1383. This indicates that the model was continuously able to improve its performance on the data it was seeing during training.

•	Validation Loss and Overfitting: The validation loss tells a more complex and crucial story. The model achieves its best validation score of 0.0618 in Epoch 8. After this point, the validation loss consistently increases for the next five epochs (Epochs 9-13) before dropping again at the end. This pattern is a classic and clear signal of overfitting. The model learned the style effectively up to Epoch 8. Beyond this point, it began to memorize the specific details of the training images, which caused its performance on the unseen validation images to degrade. This is the exact scenario that the use of a validation set is designed to detect.

## Qualitative Analysis:

Using the StableDiffusionPipeline from the diffusers library, 5 images each are generated by using the .safetensors files of both PEFT methods. For image generation, for both methods, it took ~3-5 minutes to generate an image using a prompt consisting of 77 tokens.
### Images Generated Using LoRA
![LoRA1!](ghiblivis-LORA(NEW)/BigBen.png)
![LoRA2!](ghiblivis-LORA(NEW)/Colosseum.png)
![LoRA3!](ghiblivis-LORA(NEW)/Petra.png)
![LoRA4!](ghiblivis-LORA(NEW)/BabEKhyber.png)
![LoRA5!](ghiblivis-LORA(NEW)/SydneyOperaHouse.png)

### Images Generated Using Textual Inversion
![TI1!](ghiblivis-TI(NEW)/BigBen.png)
![TI2!](ghiblivis-TI(NEW)/Colosseum.png)
![TI3!](ghiblivis-TI(NEW)/Petra.png)
![TI4!](ghiblivis-TI(NEW)/BabEKhyber.png)
![TI5!](ghiblivis-TI(NEW)/SydneyOperaHouse.png)

While quantitative metrics provide an objective measure of model performance, they cannot fully capture the nuanced, subjective qualities that define artistic style and aesthetic appeal. To address this, a qualitative user study was conducted to assess how human observers perceive and interpret the outputs of the LoRA and Textual Inversion models. This analysis is crucial for understanding the practical effectiveness of each technique in achieving the desired artistic outcome.
The study was designed as a blind comparative analysis. A panel of 50 participants was recruited. Each participant was presented with a curated set of 15 images, displayed in a randomized order. This set consisted of:

•	5 original, stylized "ghiblivis" images from the validation set.

•	5 images generated by the best-performing LoRA model.

•	5 images generated by the best-performing Textual Inversion model.

The five images from each category depicted the same five distinct landmarks to ensure a consistent basis for comparison. Crucially, participants were not informed of the origin of any image; they did not know which were original, which were generated by LoRA, or which were generated by Textual Inversion. This blind methodology is essential for eliminating potential bias and ensuring that the feedback reflects a genuine reaction to the visual qualities of the images themselves.
Participants were asked to evaluate the images based on three distinct questions, each designed to probe a different aspect of the model's performance: aesthetic appeal, style coherence, and content preservation.

A detailed analysis of the user responses reveals a clear and consistent preference for the LoRA model, while also highlighting a key challenge common to both techniques.

•	Aesthetic Appeal and Preference: The most decisive finding of the study came from the question of preference. An overwhelming 90% of participants stated that if they were to purchase an image as artwork, they would choose one generated by the LoRA model. This is a powerful indicator of LoRA's superior aesthetic quality. The feedback suggests that the LoRA-generated images were not only stylistically accurate but also more visually engaging, coherent, and possessed a "finished" quality that participants found appealing. The fact that zero participants selected a Textual Inversion image indicates a significant gap in perceived quality. The remaining 10% who would not purchase any image suggest that while the models are effective, minor artifacts or an "uncanny" quality may still be perceptible to some viewers.

•	Style Coherence and Similarity: When asked to identify which generated images were most similar in style to the original "ghiblivis" images, 80% of participants chose the LoRA set. This result strongly corroborates the quantitative findings and points to LoRA's superior ability to learn and consistently apply the target style. This is likely attributable to LoRA's mechanism; by modifying the UNet's attention layers, it learns the process and texture of the style more deeply. The resulting images likely share a more consistent color palette, brushstroke texture, and overall atmosphere that aligns closely with the training data. The 20% who found TI's output closer may have focused on specific instances where TI, by chance, replicated a particular feature well, but the overall consensus points to LoRA's greater stylistic consistency.

•	Content Preservation and Landmark Recognition: The question of landmark recognition yielded the most nuanced and critical feedback. The largest cohort, 50% of participants, felt that both models struggled to capture the defining characteristics of the real-world landmarks. This is a crucial finding, as it highlights the persistent challenge of style-content disentanglement. For both models, the aggressive application of the "ghiblivis" style appears to have interfered with the structural integrity of the content, leading to distortions or omissions of key architectural details. However, even within this challenging area, LoRA demonstrated a clear advantage. 40% of participants believed that the LoRA model did a better job of preserving the landmark's identity, compared to only 10% for Textual Inversion. This aligns with the superior LPIPS score from the quantitative analysis and suggests that LoRA's more powerful adaptation mechanism provides a better balance between applying the style and preserving the underlying content, even if it is not perfect.

## Gaps in Experimental Rigor and Optimization
The experimental setup, while functional, contained inconsistencies that impact the direct comparability of the two methods, particularly concerning efficiency.
1.	Inconsistent Optimization: The final Textual Inversion script benefited from a significant latent caching optimization, which was not implemented in the LoRA script. This led to a skewed comparison of training time; while TI was faster per-epoch, this was due to a superior implementation, not necessarily a more efficient underlying method.
2.	Initial xformers Omission: The initial TI training runs were extremely slow due to the omission of the xformers memory-efficient attention, a standard optimization that was present in the LoRA script from the start.
Improvement: For a truly fair and direct comparison, both training pipelines should be identically optimized. Both scripts should include xformers and latent caching to ensure that any observed differences in performance are attributable to the PEFT methods themselves, not the surrounding code.

## Conclusion
In every meaningful dimension of evaluation—learning stability, quantitative fidelity, and human-perceived quality—LoRA demonstrated a clear and consistent advantage. Its mechanism of directly modifying the UNet's attention layers provides a more powerful and robust means of capturing and reproducing a complex artistic style compared to the representational bottleneck of Textual Inversion's single-vector approach.

 
