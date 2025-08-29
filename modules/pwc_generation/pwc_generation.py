import numpy as np
import matplotlib.pyplot as plt
import time
from util.utils import save_output_array


from util.debug_utils import get_debug_logger
class PiecewiseCurve:
    
    def __init__(self, img, platform, sensor_info, parm_cmpd):
        self.img = img
        self.enable = parm_cmpd["is_enable"]
        self.is_debug = parm_cmpd["is_debug"]
        self.sensor_info = sensor_info
        self.bit_depth = sensor_info["bit_depth"]
        self.parm_cmpd = parm_cmpd
        self.companded_pin = parm_cmpd["companded_pin"]
        self.companded_pout = parm_cmpd["companded_pout"]
        self.is_save = parm_cmpd["is_save"]
        self.platform = platform
        # Initialize debug logger
        self.logger = get_debug_logger("PWC", config=self.platform)
    
    def generate_decompanding_lut(companded_pin, companded_pout, max_input_value=4095):
        if len(companded_pin) != len(companded_pout):
            raise ValueError("companded_pin and companded_pout must have the same length.")

        lut = np.zeros(max_input_value + 1, dtype=np.float64)

        # Handle the very first point
        lut[0:companded_pin[0]] = companded_pout[0]

        # Generate the LUT by interpolating between the knee points
        for i in range(len(companded_pin) - 1):
            start_in = companded_pin[i]
            end_in = companded_pin[i + 1]
            start_out = companded_pout[i]
            end_out = companded_pout[i + 1]

            # Use numpy for vectorized interpolation, which is much faster
            x = np.arange(start_in, end_in + 1)
            t = (x - start_in) / (end_in - start_in)
            lut[x] = start_out + t * (end_out - start_out)

        # Handle values beyond the last knee point
        last_in = companded_pin[-1]
        last_out = companded_pout[-1]
        lut[last_in:] = last_out

        return lut
    
    @staticmethod
    def generate_decompanding_curve(self):
        """
        Generates a piecewise A-Law decompanding curve to reverse the companding process.
        """
    
    # def generate_companding_curve(self):
    #     """
    #     Generates a piecewise A-Law companding curve with non-uniformly distributed kneepoints.
    #     """
    #     min_input = 0
    #     max_input = self.output_range
    #     min_output = 0
    #     max_output = self.input_range
        
    #     # Generate non-uniform kneepoints (e.g., 1, 1/2, 1/4, 1/8, etc.)
    #     x_knees = np.array([1 / (2 ** i) for i in range(self.num_kneepoints)])
    #     x_knees = np.append(x_knees, 0)
    #     x_knees = np.flip(x_knees)  # Reverse to get [1, 1/2, 1/4, ...]
        
    #     # Apply A-Law companding formula for the positive half only
    #     y_knees = np.where(
    #         x_knees == 0, 1e-6,  # If x_knees is 0, return 1e-6
    #         np.where(
    #             x_knees < (1 / self.A),  # If x_knees < (1 / self.A), return (self.A * x_knees) / (1 + np.log(self.A))
    #             (self.A * x_knees) / (1 + np.log(self.A)),
    #             (1 + np.log(self.A * x_knees)) / (1 + np.log(self.A))  # Else, return this value
    #         )
    #     )
        
    #     # Scale to the output range
    #     x_knees = x_knees * (max_input - min_input) + min_input
    #     y_knees = y_knees * (max_output - min_output) + min_output
        
    #     # Generate fine-grained x values for interpolation
    #     x_values = np.arange(min_input, max_input, 1)
    #     y_values = np.interp(x_values, x_knees, y_knees)
        
    #     return x_values, y_values, x_knees, y_knees, "Piecewise A-Law Companding Curve (Non-Uniform Kneepoints)", 'g'
    
    # import numpy as np

    # def generate_decompanding_curve(self):
    #     """
    #     Generates a piecewise A-Law decompanding curve to reverse the companding process.
    #     This function reverses the non-uniformly distributed kneepoints used in companding.
    #     """
    #     min_input = 0
    #     max_input = self.input_range  # Note: input to decompanding is output from companding
    #     min_output = 0
    #     max_output = self.output_range  # Note: output from decompanding is input to companding
        
    #     # Generate non-uniform kneepoints as in the companding function
    #     x_knees_orig = np.array([1 / (2 ** i) for i in range(self.num_kneepoints)])
    #     x_knees_orig = np.append(x_knees_orig, 0)
    #     x_knees_orig = np.flip(x_knees_orig)  # Reverse to get [0, ..., 1/4, 1/2, 1]
        
    #     # Apply A-Law companding formula for the positive half only (same as in companding)
    #     y_knees_orig = np.where(
    #         x_knees_orig == 0, 1e-6,
    #         np.where(
    #             x_knees_orig < (1 / self.A),
    #             (self.A * x_knees_orig) / (1 + np.log(self.A)),
    #             (1 + np.log(self.A * x_knees_orig)) / (1 + np.log(self.A))
    #         )
    #     )
        
    #     # Scale to the original ranges
    #     x_knees_orig = x_knees_orig * (max_output - min_output) + min_output
    #     y_knees_orig = y_knees_orig * (max_input - min_input) + min_input
        
    #     # For decompanding, we swap x and y to get the inverse function
    #     x_knees = y_knees_orig
    #     y_knees = x_knees_orig
        
    #     # Generate fine-grained x values for interpolation
    #     x_values = np.arange(min_input, max_input, 1)
    #     y_values = np.interp(x_values, x_knees, y_knees)
        
    #     # For the analytical inverse (necessary for precise decompanding)
    #     def decompand_value(y):
    #         """Analytically compute the inverse of A-Law companding for a single value"""
    #         # Scale y to [0, 1] range
    #         y_scaled = (y - min_input) / (max_input - min_input)
            
    #         # Apply inverse A-Law formula
    #         if y_scaled < (1 / self.A) / (1 + np.log(self.A)):
    #             # Inverse of: y = (A * x) / (1 + log(A))
    #             x_scaled = y_scaled * (1 + np.log(self.A)) / self.A
    #         else:
    #             # Inverse of: y = (1 + log(A * x)) / (1 + log(A))
    #             x_scaled = np.exp(y_scaled * (1 + np.log(self.A)) - 1) / self.A
            
    #         # Scale x back to original range
    #         x = x_scaled * (max_output - min_output) + min_output
    #         return x
        
    #     # Store the analytical inverse function for precise computations
    #     self.decompand_value = decompand_value
        
    #     return x_values, y_values, x_knees, y_knees, "Piecewise A-Law Decompanding Curve", 'r'  
    
    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_decompanding",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def plot_lut(self, lut, title="Decompanding LUT"):
        """
        Plot the LUT curve
        """
        plt.figure(figsize=(12, 8))
        
        # Plot the full LUT curve
        x_values = np.arange(len(lut))
        plt.plot(x_values, lut, 'b-', linewidth=2, label='LUT Curve')
        
        # Plot the knee points
        plt.plot(self.companded_pin, self.companded_pout, 'ro', markersize=8, label='Knee Points')
        
        plt.xlabel('Input Value')
        plt.ylabel('Output Value')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        filename = f"lut_plot_{self.platform['in_file']}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"LUT plot saved as: {filename}")

    def execute(self):
        self.logger.info(f"Decompanding = {self.enable}")
        if not self.enable:
            return self.img.astype(np.uint32)
        
        start = time.time()

        # Calculate the maximum possible input value for the LUT, after pedestal subtraction.
        # This is max_12bit_value - pedestal.
        max_lut_input = (2**self.bit_depth - 1) - self.parm_cmpd["pedestal"]

        # Generate the decompanding LUT using the pedestal-subtracted range.
        lut = PiecewiseCurve.generate_decompanding_lut(
            self.parm_cmpd["companded_pin"],
            self.parm_cmpd["companded_pout"],
            max_input_value=max_lut_input,
        )

        # Plot the LUT
        if self.is_debug:
            self.plot_lut(lut, f"Decompanding LUT - {self.platform['in_file']}")

        # 1. Subtract pedestal from the original image data.
        # The result must be clipped to ensure it's within the LUT's defined range.
        img_pedestal_removed = np.clip(self.img.astype(np.int64) - self.parm_cmpd["pedestal"], 0, max_lut_input)

        # 2. Apply the LUT to the pedestal-removed image.
        self.img = lut[img_pedestal_removed.astype(np.uint32)]

        self.logger.info(f"  Execution time: {time.time() - start:.3f}s")
        self.save()
        return self.img.astype(np.uint32)