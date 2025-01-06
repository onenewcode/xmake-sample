# 编译
使用 `xmake bulid` 就能直接编译

# 运行
默认情况 使用 `xmake run` 就能之间运行，但是沐曦需要占卡经行操作，默认占的卡为1，要修改显卡需要到`xmake.lua`中修改
>    os.exec("srun --gpus=1 "..binfile) -- 指定显卡