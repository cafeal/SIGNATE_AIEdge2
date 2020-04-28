PROJECT   =   centernet

CXX       :=   g++
OBJ       :=   main.o

# linking libraries of OpenCV
LDFLAGS   =   $(shell pkg-config --libs opencv)

# linking libraries of DNNDK
LDFLAGS   +=  -lhineon -ln2cube -ldputils -pthread

CUR_DIR =   $(shell pwd)
SRC     =   $(CUR_DIR)/src
BUILD   =   $(CUR_DIR)/build
VPATH   =   $(SRC)
MODEL   =   $(CUR_DIR)/dpu_centernet.elf

ARCH    =   $(shell uname -m | sed -e s/arm.*/armv71/ -e s/aarch64.*/aarch64/)
CFLAGS :=   -O2 -Wall -Wpointer-arith -std=c++11

ifeq ($(ARCH),armv71)
    CFLAGS += -mcpu=cortex-a9 -mfloat-abi=hard -mfpu=neon
endif
ifeq ($(ARCH),aarch64)
    CFLAGS += -mcpu=cortex-a53
endif

.PHONY: all clean

all: $(BUILD) $(PROJECT)

$(PROJECT) : $(OBJ)
	$(CXX) $(CFLAGS) $(addprefix $(BUILD)/, $^) $(MODEL) -o $@ $(LDFLAGS)

%.o : %.cc
	$(CXX) -c $(CFLAGS) $< -o $(BUILD)/$@

clean:
	$(RM) -rf $(BUILD)/*.o $(BUILD)
	$(RM) $(PROJECT)

$(BUILD) :
	-mkdir -p $@
