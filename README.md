# Image-Representations-and-Point-Operations
Python basic syntax and implementation of some of its image processing facilities (Histogram equalization, Optimal image quantization, Gamma Correction, etc.).

-----

1. [ Image Reading & Display ](#image-reading--display)
2. [ Conversion of RGB to YIQ (and vice versa) ](#conversion-of-rgb-to-yiq-and-vice-versa)
3. [ Hsitogram Equalize ](#hsitogram-equalize)
	- [ Gray ](#-gray-)
	- [ RGB ](#-rgb-)
4. [ Quantize Image ](#quantize-image)
	- [ Gray (3 Colors example) ](#-gray-3-colors-)
	- [ RGB (3 Colors example) ](#-rgb-3-colors-)
5. [ Gamma Tool ](#gamma-tool)

-----
  
  
     
<h2>Image Reading & Display</h2>

| Read As RGB (Original) | Read As Grayscale |
| ------------- | ------------- |
| <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/colored.png"/></p>  | <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/gray.png"/></p>  |

-----

<h2>Conversion of RGB to YIQ (and vice versa)</h2>

<div align="center">
  
| RGB to YIQ (and vice versa) |
| ------------- |
| <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/rgb_yiq.png"/></p>  |
  
</div>

-----

<h2>Hsitogram Equalize</h2>

<h3> Gray: </h3>

| Hsitogram Equalize (Gray) | Comparison |
| ------------- | ------------- |
| <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/hist_gray.png"/></p>  | <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/hist_gray_change.png"/></p>  |

<h3> RGB: </h3>

| Hsitogram Equalize (RGB) | Comparison |
| ------------- | ------------- |
| <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/hist_colored.png"/></p>  | <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/hist_colored_change.png"/></p>  |

-----

<h2>Quantize Image</h2>

<h3> Gray, 3 Colors: </h3>

| Quantize Image (Gray, 3 Colors) | After minimizing the error | Error measurement |
| ------------- | ------------- | ------------- |
| <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/equalize_three_colors.png"/></p>  | <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/equalize_three_colors_corrected.png"/></p>  | <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/equalize_three_colors_change.png"/></p>  |

<h3> RGB, 3 Colors: </h3>

| Quantize Image (RGB Intensity, 3 Colors) | After minimizing the error | Error measurement |
| ------------- | ------------- | ------------- |
| <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/equalize_yiq_three_colors.png"/></p>  | <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/equalize_yiq_three_colors_corrected.png"/></p>  | <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/equalize_yiq_three_colors_change.png"/></p>  |

-----

<h2>Gamma Tool</h2>

| Gamma Tool (0.85 VALUE) | Gamma Tool (1.1 VALUE) | Gamma Tool (2.0 VALUE) |
| ------------- | ------------- | ------------- |
| <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/gamma_85.jpg"/></p>  | <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/gamma_110.jpg"/></p>  | <p align="center"><img src="https://github.com/AlmogJakov/Image-Representations-and-Point-Operations/blob/main/demo/gamma_200.jpg"/></p>  |

-----
