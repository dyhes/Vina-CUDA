# 编译器
CXX := g++

# 编译器选项
CXXFLAGS := -std=c++17

# 源文件目录和目标文件目录
SRCDIR := ./
OBJDIR := obj

# 查找所有的 .cpp 文件
SRC := $(wildcard $(SRCDIR)/*.cpp)

# 将 .cpp 文件对应为 .o 文件
OBJ := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRC))

# 最终生成的可执行文件
TARGET := program

# 默认规则
all: $(TARGET)

# 链接目标文件生成最终的可执行文件
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# 编译 .cpp 文件生成 .o 文件
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 创建目标文件目录
$(OBJDIR):
	mkdir -p $(OBJDIR)

# 清理编译生成的文件
clean:
	rm -rf $(OBJDIR) $(TARGET)

