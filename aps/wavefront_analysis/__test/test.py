

from aps.wavefront_analysis.driver.wavefront_sensor import WavefrontSensor, get_default_file_name_prefix, get_image_file_data

wf = WavefrontSensor()

print(get_image_file_data(measurement_directory="/home/beams/S3BLUE/Documents/AI-Autoalignment/Working-Directories/3-ID/TEST_WF/wf_images",
                          file_name_prefix=get_default_file_name_prefix(),
                          image_index=1))

