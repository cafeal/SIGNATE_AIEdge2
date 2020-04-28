DEBUG = 0

# 01 freeze graph
KERAS_MODEL = model.h5
COPIED_KERAS_MODEL = keras_model.h5
GPU = "0"
RESULT_01 = result_01

# 02 quantize
FROZEN_GRAPH = frozen_graph.pb
QUANT_GRAPH = deploy_model.pb
INPUT_NODE = image
OUTPUT_NODE = detector_model/output/BiasAdd
CALIB_ITER = 100
BATCH_SIZE = 8
SCALE = 4
RESULT_02 = result_02

# 03 compile
DPU_NET_NAME = centernet
OUTPUT_ELF = dpu_$(DPU_NET_NAME).elf
ELF_FILE = $(OUTPUT_ELF)
DCF_FILE = qiita.dcf
ARCH_FILE = ultra96v2_oob.json
RESULT_03 = result_03

VAI_Q = /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/bin/vai_q_tensorflow
VAI_C = /opt/vitis_ai/conda/envs/vitis-ai-tensorflow/bin/vai_c_tensorflow

USER = `whoami`
UID = `id -u`
GID = `id -g`
HERE = `pwd`

ifeq ($(DEBUG),1)
	CALIB_ITER = 5
endif

ifeq ($(SCALE),4)
	INPUT_SHAPES = ?,320,512,3
endif
ifeq ($(SCALE),8)
	INPUT_SHAPES = ?,160,256,3
endif



.PHONY: all clean

all: $(ELF_FILE)
	

$(ELF_FILE): $(QUANT_GRAPH) $(ARCH_FILE)
	docker run \
		-v /opt/xilinx/dsa:/opt/xilinx/dsa \
		-v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
		-e USER=$(USER) -e UID=$(UID) -e GID=$(GID) \
		-v $(HERE):/workspace \
		-v $(HERE)/vai_c_tensorflow:/opt/vitis_ai/conda/envs/vitis-ai-tensorflow/bin/vai_c_tensorflow \
		-w /workspace \
		-ti \
		--rm \
		--runtime=nvidia \
		--network=host \
		xilinx/vitis-ai:tools-1.0.0-gpu \
		$(VAI_C) \
			--frozen_pb $(QUANT_GRAPH) \
			--arch $(ARCH_FILE) \
			--output_dir $(RESULT_03) \
			--net_name $(DPU_NET_NAME) \
	&& cp $(RESULT_03)/$(OUTPUT_ELF) $(ELF_FILE)

$(ARCH_FILE):
	echo '{"target":"dpuv2","dcf":"$(DCF_FILE)","cpu_arch":"arm64"}' > $(ARCH_FILE)

$(QUANT_GRAPH): $(FROZEN_GRAPH)
	docker run \
		-v /opt/xilinx/dsa:/opt/xilinx/dsa \
		-v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
		-e USER=$(USER) -e UID=$(UID) -e GID=$(GID) \
		-e PYTHONPATH=/opt/vitis_ai/compiler \
		-v $(HERE):/workspace \
		-w /workspace \
		--rm \
		--runtime=nvidia \
		--network=host \
		xilinx/vitis-ai:tools-1.0.0-gpu \
		INPUT_NODE=$(INPUT_NODE) \
		BATCH_SIZE=$(BATCH_SIZE) \
		SCALE=$(SCALE) \
		$(VAI_Q) quantize \
			--input_frozen_graph $(FROZEN_GRAPH) \
			--input_nodes $(INPUT_NODE) \
			--input_shapes $(INPUT_SHAPES) \
			--output_nodes $(OUTPUT_NODE) \
			--input_fn input_fn.calib_input \
			--method 1 \
			--gpu $(GPU) \
			--calib_iter $(CALIB_ITER) \
			--output_dir $(RESULT_02) \
	&& cp $(RESULT_02)/$(QUANT_GRAPH) $(QUANT_GRAPH)

$(FROZEN_GRAPH): $(COPIED_KERAS_MODEL)
	cd ../../ \
	&& docker-compose run --rm \
		-e NVIDIA_VISIBLE_DEVICES=$(GPU) \
		app bash -c " \
			cd deploy/convert \
			&& python3 01_freeze_graph.py \
		    	-m $(COPIED_KERAS_MODEL) \
				-g $(FROZEN_GRAPH) \
				-o $(RESULT_01) \
			&& chown -R $(UID):$(GID) $(RESULT_01) \
		" \
	&& cd deploy/convert \
	&& cp $(RESULT_01)/$(FROZEN_GRAPH) $(FROZEN_GRAPH)

$(COPIED_KERAS_MODEL): $(KERAS_MODEL)
	cp $(KERAS_MODEL) $(COPIED_KERAS_MODEL)

clean:
	rm -rf $(RESULT_01) $(RESULT_02) $(RESULT_03)
	rm -f $(COPIED_KERAS_MODEL)
	rm -f $(ARCH_FILE)
	rm -f frozen_graph.pb
	rm -f deploy_model.pb