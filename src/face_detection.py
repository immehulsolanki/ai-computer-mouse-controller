"""computer pointer controller"""
"""
Copyright [2020] [MEHUL SOLANKI]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

""" Model Information """
"""
This class implemented to work with face dection model. Refere following parameters to integrate or modify it.

Name: face-detection-adas-binary-0001

Specification
METRIC	VALUE
AP (head height >10px)	31.2%
AP (head height >32px)	76.2%
AP (head height >64px)	90.3%
AP (head height >100px)	91.9%
Min head size	90x90 pixels on 1080p
GFlops	0.611
GI1ops	2.224
MParams	1.053
Source framework	PyTorch*

Inputs
Name: input, shape: [1x3x384x672] - An input image in the format [BxCxHxW], where:

B - batch size
C - number of channels
H - image height
W - image width
Expected color order is BGR.

Outputs
The net outputs blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes. Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max], where:

image_id - ID of the image in the batch
label - predicted class ID
conf - confidence for the predicted class
(x_min, y_min) - coordinates of the top left bounding box corner
(x_max, y_max) - coordinates of the bottom right bounding box corner.

Legal Information
ref: https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html
"""
import os
import sys
import logging as log
import cv2
from openvino.inference_engine import IENetwork, IECore

class face_detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model = model
        self.device = device
        self.extension = extensions
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None
        self.device_list = []
        self.print_once = True
        self.network_input_shape = None

        # ====== Load model files, verify model, and get the input shap parameters at start
        self.load_model()
        self.check_model()
        self.network_input_shape = self.get_input_shape()

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IECore()

        log.info("------device avaibility--------")
        for available_devices in self.plugin.available_devices: #Dont use device variable, conflicts.
            self.device_list.append(available_devices)
        log.info("Available device: "+ str(self.device_list)) #get name of available devices

        log.info("---------Plugin version--------")
        ver = self.plugin.get_versions("CPU")["CPU"] # get plugin info
        log.info("descr: maj.min.num"+ str(ver.description) +"."+ str(ver.major) +"." + str(ver.minor)+"." + str(ver.build_number))

        ### Load IR files into their related class
        model_xml = self.model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Check if path is not given or model name doesnt ends with .xml
        if model_xml is not None and model_xml.find('.xml') != -1:
            f,s = model_xml.rsplit(".",1) #check from last "." and "r"split only one element from last
            model_bin = f + ".bin"
        else:
            log.error("Error! Model files are not found or invalid, check paths")
            log.error("Program stopped")
            sys.exit() #exit program no further execution
        log.info("-------------Model path----------")
        log.info("XML: "+ str(model_xml))
        log.info("bin: "+ str(model_bin))

        self.network = IENetwork(model=model_xml, weights=model_bin)
        log.info("ModelFiles are successfully loaded into IENetwork")

        return 

    def check_model(self):
        '''
        This function will check the input model for compatibility with hardware, 
        It will check for supported and unsuported layers if any.
        No return required.
        '''
        ### Check for supported layers ###
        log.info("Checking for supported Network layers...")
        # Query network will return all the layer, required all the time if device changes.
        supported_layers = self.plugin.query_network(network=self.network, device_name="CPU")
        log.info("------Status of default Network layers--------")
        log.info("No. of Layers in network: "+ str(len(self.network.layers)))
        log.info("No. of supported layers:"+ str(len(supported_layers)))

        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            log.info("Unsupported layers found:"+ str(unsupported_layers))
            log.info("CPU extension required and adding...")
            #exit(1)
        ### Adding any necessary extensions ###
            if self.extension and "CPU" in self.device:
                self.plugin.add_extension(self.extension, self.device)
                log.info("Checking for CPU extension compatibility...")
                # Again Query network will return fresh list of supported layers.
                supported_layers = self.plugin.query_network(network=self.network, device_name="CPU")
                log.info("------Status of Network layers with CPU Extension--------")
                log.info("No. of Layers in network:"+ str(len(self.network.layers)))
                log.info("No. of supported layers:"+ str(len(supported_layers)))
                log.info("CPU extension added sucessfully!")

                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers) != 0:
                    log.error("Unsupported layers found: "+ str((unsupported_layers)))
                    log.error("Error! Model not supported, Program stopped")
                    exit()
            else:
                log.error("Error! cpu extension not found")
                log.error("Program stopped")
                exit()
        else:
            log.info("All the layers are supported, No CPU extension required")

        # This will enable all following four functions ref:intel doc. ie_api.IECore Class Reference
        self.exec_network = self.plugin.load_network(self.network, "CPU")
        print("IR successfully loaded into Inference Engine")
        
        return

    def get_input_shape(self):
        '''
        Get dimensions of inputs
        '''
        ### Get the input layer informations ###
        log.info("-----Accessing input layer information-----")
        log.info('Face detection model Network input layers = ' + str(list(self.network.inputs)))
        log.info('Face detection model Network input layers type: '+ str(type(self.network.inputs)))
        self.input_blob = next(iter(self.network.inputs))#Origional
        log.info("-------------------------------")
        return self.network.inputs[self.input_blob].shape #Origional

    def wait(self):
        ### Wait for the Async request to be complete. ###
        status = self.exec_network.requests[0].wait(-1)
        return status

    def get_output(self):
        ### Extract and return the output results
        self.output_blob = next(iter(self.network.outputs))
        # First return the name of blob as dictionary and second output of first blob as Nd array
        return self.exec_network.requests[0].outputs, self.exec_network.requests[0].outputs[self.output_blob]

    def preprocess_input(self, input_frames_raw, network_input_shape_height, network_input_shape_width):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_frame = cv2.resize(input_frames_raw, (network_input_shape_height, network_input_shape_width)) #Resize as per network input spec.
        p_frame = p_frame.transpose((2,0,1)) #swap channel cxhxw 
        p_frame = p_frame.reshape(1, *p_frame.shape) #add one axis 1 to make 4D shape for network input
        #print(p_frame.shape) #Debug output
        return p_frame

    def preprocess_output(self, input_frames_raw, result, input_frames_raw_width, input_frames_raw_height, prob_threshold):
        '''
        Draw bounding boxes onto the frame.
        '''
        countmultipeople = 0
        for box in result[0][0]: # Output shape is 1x1x100x7
            conf = box[2]
            if conf >= prob_threshold:
                countmultipeople += 1
                xmin = int(box[3] * input_frames_raw_width)
                ymin = int(box[4] * input_frames_raw_height)
                xmax = int(box[5] * input_frames_raw_width)
                ymax = int(box[6] * input_frames_raw_height)
                # label = "Person"+str(countmultipeople)
                cv2.rectangle(input_frames_raw, (xmin, ymin), (xmax, ymax), (0,0,255), 1) #main rect.
                # cv2.rectangle(input_frames_raw, (xmin, ymin), (xmin+90, ymin+10), (0,0,255), -1) # Text rect.
                # cv2.putText(input_frames_raw, label, (xmin,ymin+10),cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,255), 1)
        return input_frames_raw, countmultipeople

    def predict(self, input_frames_raw, input_frame_raw_height, input_frame_raw_width, prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image or video frame.
        input: RBG image in jpg format or video frame.
        '''
        p_frame = self.preprocess_input(input_frames_raw, self.network_input_shape[3], self.network_input_shape[2])
        self.exec_network.start_async(request_id=0, #Origional
                inputs={self.input_blob: p_frame}) # run inference
        if self.wait() == 0:
            ### the results of the inference request ###
            blob, result = self.get_output()

            # Print available blob infirmation
            blob_list = []
            if self.print_once: # Print only Once
                self.print_once = False
                for name, output_ in blob.items(): #Find the possible BLOBS for name, 
                    blob_list.append(name)
                log.info("The name of available blob is: " + str(blob_list))
                
            p_frame, countmultipeople = self.preprocess_output(input_frames_raw, result, input_frame_raw_height, input_frame_raw_width, prob_threshold)
            return p_frame, countmultipeople