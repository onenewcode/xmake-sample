toolchain("muxi.toolchain")
    set_kind("standalone")
    set_toolset("ld", "nvcc@mxcc")
    set_toolset("cxx", "nvcc@mxcc")
    on_load(function (toolchain)
        toolchain:add("cxflags", "-x", "maca")
        if not is_plat("windows")  then
            for _, includedir in ipairs({"/opt/maca/include", "/opt/maca/mxgpu_llvm/include"}) do
                if os.isdir(includedir) then
                    toolchain:add("includedirs", includedir)
                end
            end
            for _, linkdir in ipairs({"/opt/maca/lib", "/opt/maca/mxgpu_llvm/lib"}) do
                if os.isdir(linkdir) then
                    toolchain:add("linkdirs", linkdir)
                end
            end
        end
    end)
target("muxi")
    set_kind("binary")    -- 目标将编译为二进制可执行文件
    add_files("src/*.cpp")
    set_toolchains("muxi.toolchain") -- 设置工具链
target_end()
