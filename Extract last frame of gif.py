"""
Extract the final frame of a graphic interchange file

"""

from PIL import Image

def extract_last_frame(gif_path, output_path):
    with Image.open(gif_path) as gif:
        # Seek to the last frame
        try:
            while True:
                gif.seek(gif.tell() + 1)
        except EOFError:
            # Now at the last frame
            pass
        
        # Save last frame
        gif.save(output_path)

# Example usage:
extract_last_frame("dendrite_growth_simulation-pulses-newcurrent-20_electrons-visc-2.5_vdW-2.0.png")
