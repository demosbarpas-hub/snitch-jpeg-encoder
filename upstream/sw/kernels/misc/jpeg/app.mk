# Copyright 2026 Dimosthenis Karamparpas
# SPDX-License-Identifier: Apache-2.0

APP              := jpeg
$(APP)_BUILD_DIR ?= $(SN_ROOT)/sw/kernels/misc/$(APP)/build
SRC_DIR          := $(SN_ROOT)/sw/kernels/misc/$(APP)/src
SRCS             := $(SRC_DIR)/main.c

include $(SN_ROOT)/sw/kernels/datagen.mk
include $(SN_ROOT)/sw/kernels/common.mk
