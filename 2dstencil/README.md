# Explanation:

 EBISU is based on a simple principle:
 - Restrict parallelism and then
 - Ccale the usage of resources

 # Usage:
  run ./config.sh 
  
  run ./build.sh
  
  the built executable files is generated in ./build/init/
  
  # RUN parameter:
  *ebisu.exe domainwidthy domaindwidthx [options]
  
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
   - Temporal blocking kernel: _jacobi-temporal-traditional-box.cu_ and _jacobi-temporal-traditional.cu_
   - Main: _jacobi.driver.cpp_

# Output
ptx_version SizeofData width_x width_y iteration depth bdim valid_bdim  <gridx, gridy, gridz> block_per_sm shared_mem_used shared_mem_range runtime(ms) performance(GCells/s)

Use verbose for detailed output
