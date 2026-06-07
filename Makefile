CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -march=native
LDFLAGS = -lm

SRC     = src
BUILD   = build
TARGET  = $(BUILD)/cml
SRCS    = main.c tensor_core.c tensor_math.c tensor_nn.c tensor_loss.c activation.c
OBJS    = $(addprefix $(BUILD)/, $(SRCS:.c=.o))

.PHONY: all clean debug

all: $(BUILD) $(TARGET)

$(BUILD):
	mkdir -p $(BUILD)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD)/%.o: $(SRC)/%.c
	$(CC) $(CFLAGS) -I$(SRC) -c -o $@ $<

debug: CFLAGS = -Wall -Wextra -g -O0 -fsanitize=address,undefined
debug: LDFLAGS += -fsanitize=address,undefined
debug: $(BUILD) $(TARGET)

clean:
	rm -rf $(BUILD)
