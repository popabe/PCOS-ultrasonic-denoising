import torch
from diffusers import DDPMScheduler, UNet2DModel

class DiffusionModel:
    def __init__(self, latents_size=256, layers_per_block=2, device='cuda'):
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='linear')
        self.create_model(latents_size, layers_per_block, device)

    def create_model(self, latents_size, layers_per_block, device):
        self.model = UNet2DModel(
            sample_size=latents_size,
            in_channels=2,
            out_channels=1,
            layers_per_block=layers_per_block,
            block_out_channels=[128, 256, 512, 512],
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
        )
        self.model.to(device=device)

    def load_model(self, path, device):
        self.model = torch.load(path)
        self.model.to(device)

    def get_model(self):
        return self.model

    def initialize_opt_loss_function(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00005)
        self.loss_function = torch.nn.functional.mse_loss

    def add_noise(self, images, noise, timesteps):
        return self.noise_scheduler.add_noise(images, noise, timesteps)

    def denoise_images(self, low_res_images, device):
        latents = torch.rand_like(low_res_images, device=low_res_images.device)
        for t in self.noise_scheduler.timesteps:
            with torch.no_grad():
                latents_input = torch.cat([latents, low_res_images], dim=1)
                noise_preds = self.model(latents_input, t).sample
                latents = self.noise_scheduler.step(noise_preds, t, latents).prev_sample
        return latents, noise_preds

    def train_model(self, dataset, test_dl, epochs, verbose, device, batch_size, latent_size, starting_epoch=1):
        import pickle
        import numpy as np
        import matplotlib.pyplot as plt
        import torchvision

        losses = []
        for epoch in range(starting_epoch, epochs + 1):
            train_loss = 0
            for step, (img_noisy, img_clean) in enumerate(dataset):
                img_clean = img_clean.to(device)
                img_noisy = img_noisy.to(device)
                self.optimizer.zero_grad()
                timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (batch_size,), device=device)
                noise = torch.randn((batch_size, 1, latent_size, latent_size), device=device)
                noisy_latents = self.noise_scheduler.add_noise(img_clean, noise=noise, timesteps=timesteps)
                noise_prediction = self.model(torch.cat([noisy_latents, img_noisy], dim=1), timesteps)[0]
                loss = self.loss_function(noise_prediction, noise)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_size
            train_loss = train_loss / len(dataset)
            losses.append(train_loss)
            with open('diffusion_train_loss.pkl', 'wb') as output:
                pickle.dump(losses, output)
            print(f'Epoch: {epoch} / loss:{train_loss:.3f}')
            if (epoch + 1) % verbose == 0:
                self.model.eval()
                test_imgs, test_true = next(iter(test_dl))
                test_imgs = test_imgs.to(device)
                with torch.no_grad():
                    denoised_images, noise_preds = self.denoise_images(test_imgs, device)
                plt.figure(figsize=(15, 10))
                plt.subplot(1, 3, 1)
                plt.axis("off")
                plt.title("Noisy Inputs")
                plt.imshow(np.transpose(torchvision.utils.make_grid(test_imgs, nrow=2, padding=1, normalize=True).cpu(), (1, 2, 0)))
                plt.subplot(1, 3, 2)
                plt.axis("off")
                plt.title("Denoised Results")
                plt.imshow(np.transpose(torchvision.utils.make_grid(denoised_images.detach().cpu(), nrow=2, padding=1, normalize=True), (1, 2, 0)))
                plt.subplot(1, 3, 3)
                plt.axis("off")
                plt.title("Predicted Noise")
                plt.imshow(np.transpose(torchvision.utils.make_grid(noise_preds.detach().cpu(), nrow=2, padding=1, normalize=True), (1, 2, 0)))
                plt.show()
                plt.clf()
