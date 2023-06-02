# Explanation:

 EBISU based on a sinple principle:
 - Restrict parallelism and then scaling the usage of resources

 # Usage:
  run ./config.sh 
  
  run ./build.sh
  
  the builded excutable files is generated in ./build/init/
  
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
