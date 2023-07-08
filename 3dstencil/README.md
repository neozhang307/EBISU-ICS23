# Explanation:

 EBISU is based on a simple principle:
 - Constraining parallelism
 - Scaline the usage of resources

 # Usage:
  run ./config.sh 
  
  run ./build.sh
  
  the built executable files are generated in ./build/init/
  
  # RUN parameter:
  *ebisu.exe domaindwidthx domainwidthy domainwidthz [options]
  
--fp32            (single precision)

--check           (check result with CPU)

--warmup          (set warmup run)

--verbose         (details the output)

--experiment      (Exeperiment setting for ebisu in ICS23)

--bdim=[128,256]  (block dimemtion)

--blkpsm=[TB per SMX]

--iter=[total time step]

  # RUN EBISU setting:
   ./*ebisu.exe --experiment

  # Important Files
   - Temporal Blocking Depth: _common/temporalconfig.cuh_
   - Parallelism: _common/iptconfig.cuh_
   - Core MCQ file: _common/multi_queue.cuh_
   - Temporal blocking kernel: _j3d-temporal-box.cu_ and _j3d-temporal.cu_
   - Main: _j3d.driver.cpp_

# Output
ptx_version SizeofData width_x width_y height iteration depth bdim ilp <gridx, gridy, gridz> block_per_sm shared_mem_used shared_mem_range runtime(ms) performance(GCells/s)
