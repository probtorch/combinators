import torch
import torch.nn as nn
from torch.nn.functional import affine_grid, grid_sample

"""
Affine Tranformer that either transfers a frame to a digit image or transfers a digit to a frame
==========
abbreviations:
K -- number of digits
T -- timesteps in one bmnist sequence
S -- sample size
B -- batch size
==========
function contracts
affine_grid https://pytorch.org/docs/1.3.0/nn.functional.html?highlight=affine_grid#torch.nn.functional.affine_grid
    affine matrices: N * 2 * 3, torch.Size((N, C, H_out, W_out))
    ===> affine grid: N * C * H_out * W_out
grid_sample https://pytorch.org/docs/1.3.0/nn.functional.html?highlight=affine_grid#torch.nn.functional.grid_sample
    input images: N * H_in, H_in, affine grid: N * C * H_out * W_out
    ===> output images: N * C * H_out * W_out
!!!Note that these two functions will be overwritten in pytorch 1.4.0!!!
==========
"""
class Affine_Transformer():
    def __init__(self, frame_pixels, digit_pixels, device):
        """
        scale_dtof, translation_dtof: scaling and translation factors in transformation from digit to frame
        scale_ftod, translation_ftod: scaling and translation factors in transformation from frame to digit
        """
        super().__init__()
        self.digit_pixels =  digit_pixels
        self.frame_pixels = frame_pixels
        self.translation_dtof = (self.frame_pixels - self.digit_pixels) / self.digit_pixels
        self.translation_ftod = (self.frame_pixels - self.digit_pixels) / self.frame_pixels
        self.scale_dtof = torch.FloatTensor([[self.frame_pixels / self.digit_pixels, 0], [0, self.frame_pixels / self.digit_pixels]])
        self.scale_ftod = torch.FloatTensor([[self.digit_pixels / self.frame_pixels, 0], [0, self.digit_pixels / self.frame_pixels]])
        if torch.cuda.is_available():
            self.scale_dtof = self.scale_dtof.to(device)
            self.scale_ftod = self.scale_ftod.to(device)

    def digit_to_frame(self, digit, z_where):
        """
        transfer the digits to the frame
        [digit: S * B * K * DP * DP, z_where: S * B * T * K * 2 ===> frame: S * B * T * K * FP * FP]
        """
        S, B, T, K, _ = z_where.shape
        affine_p1 = self.scale_dtof.repeat(S, B, T, K, 1, 1)## S * B * T * K * 2 * 2
        affine_p2 = z_where.unsqueeze(-1) * self.translation_dtof ## S * B * T * K * 2 * 1
        affine_p2[:, :, :, :, 0, :] = -1 * affine_p2[:, :, :, :, 0, :] ## flip the x-axis due to grid function
        grid = affine_grid(torch.cat((affine_p1, affine_p2), -1).view(S*B*T*K, 2, 3), torch.Size((S*B*T*K, 1, self.frame_pixels, self.frame_pixels)), align_corners=True)
        frames = grid_sample(digit.unsqueeze(2).repeat(1,1,T,1,1,1).view(S*B*T*K, self.digit_pixels, self.digit_pixels).unsqueeze(1), grid, mode='nearest', align_corners=True)
        return frames.squeeze(1).view(S, B, T, K, self.frame_pixels, self.frame_pixels)

    def frame_to_digit(self, frames, z_where):
        """
        transfer the frames to the digits
        [frame: S * B * T * FP * FP, z_where: S * B * T * K * 2 ===> digit: S * B * T * K * DP * DP]
        """
        S, B, T, K, _ = z_where.shape
        affine_p1 = self.scale_ftod.repeat(S, B, T, K, 1, 1)## S * B * T * K * 2 * 2
        affine_p2 = z_where.unsqueeze(-1) * self.translation_ftod ## S * B * T * K * 2 * 1
        affine_p2[:, :, :, :, 1, :] = -1 * affine_p2[:, :, :, :, 1, :] ## flip the y-axis due to grid function
        grid = affine_grid(torch.cat((affine_p1, affine_p2), -1).view(S*B*T*K, 2, 3), torch.Size((S*B*T*K, 1, self.digit_pixels, self.digit_pixels)), align_corners=True)
        digit = grid_sample(frames.unsqueeze(-3).repeat(1, 1, 1, K, 1, 1).view(S*B*T*K, self.frame_pixels, self.frame_pixels).unsqueeze(1), grid, mode='nearest', align_corners=True)
        return digit.squeeze(1).view(S, B, T, K, self.digit_pixels, self.digit_pixels)
