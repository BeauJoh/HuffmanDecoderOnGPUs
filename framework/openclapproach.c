#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <signal.h>
#include <assert.h>
#include <unistd.h>
#include <stdbool.h>

void end (void) __attribute__((destructor));

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "decodeUtil.h"
#include "timing.h"

double build_time, buffer_creation_time, memory_movement_time, context_time;

/*
 * OpenCL Utility Functions
 */
cl_kernel getKernelFromSource(const char* source_path,
                              const char* kernel_name,
                              cl_device_id device,
                              cl_context context) {

    struct timer *t;
	t = newTimer();
    timestart(t);

    cl_int error;
    FILE* file_handle = fopen(source_path, "r");
    fseek(file_handle,0,SEEK_END);
    size_t kernel_source_length = ftell(file_handle);
    char* kernel_source = (char*)malloc(sizeof(char)*(kernel_source_length+1));
    rewind(file_handle);
    size_t items_read = fread(kernel_source,        //ptr
                              sizeof(char),         //size
                              kernel_source_length, //count
                              file_handle);         //stream
    fclose(file_handle);

    assert(items_read == kernel_source_length &&
           "Failed to read source file");

    kernel_source[kernel_source_length] = '\0';
    cl_program program =
        clCreateProgramWithSource(context,                        //context
                                  1,                              //count
                                  ((const char **)&kernel_source),//string
                                  &kernel_source_length,          //lengths
                                  &error);                        //errcode_ret
    assert(!error && "Failed to create program from source");
#if defined(DEBUG) && defined(CPU_BUILD)
    char cwd[1024];
    char* result = getcwd(cwd,sizeof(cwd));
    assert(result != NULL);
    char options[strlen("-g -s ")+strlen(cwd)+strlen("/")+strlen(source_path)];
    sprintf(options, "-g -s %s/%s", cwd, source_path);
#elif GPU_BUILD
    char* options = "-cl-mad-enable -cl-fast-relaxed-math"
#else
    char* options = NULL;
#endif
    error = clBuildProgram(program, //program
                           1,       //num_devices
                           &device, //device_list
                           (const char*)options, //compiler options
                           NULL,    //pfn_notify
                           NULL);   //user_data
    assert(!error && "Failed to build program");

    cl_kernel kernel = clCreateKernel(program,    //program
                                      kernel_name,//kernel_name
                                      &error);    //errcode_ret
    assert(!error && "Failed to create kernel");

    free(kernel_source);
    clReleaseProgram(program);

    timestop(t);
    build_time += timerms(t);

    return kernel;
}

bool loadKernelFromSourceAndSaveAsBinary(const char* source_path,
                                         const char* binary_path,
                                         cl_device_id device,
                                         cl_context context) {

    cl_int error;
    FILE* file_handle = fopen(source_path, "r");
    fseek(file_handle,0,SEEK_END);
    size_t kernel_source_length = ftell(file_handle);
    char* kernel_source = (char*)malloc(sizeof(char)*(kernel_source_length+1));
    rewind(file_handle);
    size_t items_read = fread(kernel_source,        //ptr
                              sizeof(char),         //size
                              kernel_source_length, //count
                              file_handle);         //stream
    fclose(file_handle);

    assert(items_read == kernel_source_length &&
           "Failed to read source file");

    kernel_source[kernel_source_length] = '\0';
    cl_program program =
        clCreateProgramWithSource(context,                        //context
                                  1,                              //count
                                  ((const char **)&kernel_source),//string
                                  &kernel_source_length,          //lengths
                                  &error);                        //errcode_ret
    assert(!error && "Failed to create program from source");
#if defined(DEBUG) && defined(CPU_BUILD)
    char cwd[1024];
    char* result = getcwd(cwd,sizeof(cwd));
    assert(result != NULL);
    char options[strlen("-g -s ")+strlen(cwd)+strlen("/")+strlen(source_path)];
    sprintf(options, "-g -s %s/%s", cwd, source_path);
#elif GPU_BUILD
    char* options = "-cl-mad-enable -cl-fast-relaxed-math"
#else
    char* options = NULL;
#endif
    error = clBuildProgram(program, //program
                           1,       //num_devices
                           &device, //device_list
                           (const char*)options, //compiler options
                           NULL,    //pfn_notify
                           NULL);   //user_data
    assert(!error && "Failed to build program");

    size_t kernel_length;
    error = clGetProgramInfo(program,                //program
                             CL_PROGRAM_BINARY_SIZES,//param name
                             sizeof(size_t),         //param value size
                             &kernel_length,         //param value
                             NULL);                  //param value return size
    assert(!error && "Failed to get the length of the kernel");

    unsigned char* program_binary =
        (unsigned char*) malloc(sizeof(unsigned char)*kernel_length);
    error = clGetProgramInfo(program,                //program
                             CL_PROGRAM_BINARIES,    //param name
                             kernel_length,          //param value size
                             &program_binary,        //param value
                             NULL);                  //param value return size
    assert(!error && "Failed to get the kernel binary");

    file_handle = fopen(binary_path, "wb");
    size_t items_writ = fwrite(program_binary,          //ptr
                               sizeof(unsigned char),   //size
                               kernel_length,           //count
                               file_handle);            //stream
    assert(items_writ == kernel_length && "Failed to write binary file");
    fclose(file_handle);
    
    free(kernel_source);
    free(program_binary);
    clReleaseProgram(program);
    return true;
}

cl_kernel getKernelFromBinary(const char* binary_path,
                              const char* kernel_name,
                              cl_device_id device,
                              cl_context context) {

    struct timer *t;
	t = newTimer();
    timestart(t);

    cl_int error;
    FILE* file_handle = fopen(binary_path, "r");
    fseek(file_handle,0,SEEK_END);
    size_t kernel_binary_length = ftell(file_handle);
    unsigned char* kernel_binary =
        (unsigned char*)malloc(sizeof(unsigned char)*(kernel_binary_length));
    rewind(file_handle);
    size_t items_read = fread(kernel_binary,            //ptr
                              sizeof(unsigned char),    //size
                              kernel_binary_length,     //count
                              file_handle);             //stream
    fclose(file_handle);
    
    assert(items_read == kernel_binary_length &&
           "Failed to read binary file");
    cl_int binary_status;
    cl_program program = clCreateProgramWithBinary(
            context,                                    //context
            1,                                          //num devices
            &device,                                    //device list
            &kernel_binary_length,                      //lengths
            ((const unsigned  char **)&kernel_binary),  //binaries
            &binary_status,                             //binary status
            &error);                                    //errcode_ret
    assert(!error && "Failed to create program from binary");
    assert(binary_status == CL_SUCCESS && "The program binary was not loaded");

    error = clBuildProgram(program, //program
                           1,       //num_devices
                           &device, //device_list
                           NULL,    //compiler options
                           NULL,    //pfn_notify
                           NULL);   //user_data
    assert(!error && "Failed to build program");

    cl_kernel kernel = clCreateKernel(program,    //program
                                      kernel_name,//kernel_name
                                      &error);    //errcode_ret
    assert(!error && "Failed to create kernel");

    free(kernel_binary);
    clReleaseProgram(program);

    timestop(t);
    build_time += timerms(t);

    return kernel;
}

/*
 * End of OpenCL Utility Functions
 */

cl_platform_id static platform = NULL;
cl_device_id static device = NULL;
cl_context static context = NULL;
cl_command_queue static queue = NULL;

void openclApproach(struct CompressedData *cd,
		            struct UnCompressedData *uncompressed,
                    void *paramdata)   {
    //initialise timing results
    build_time = 0.0;
    buffer_creation_time = 0.0;
    memory_movement_time = 0.0;
    context_time = 0.0;

    //buffer creation timer
    struct timer *bct;
    bct = newTimer(); 

    //memory movement timer
    struct timer *mmt;
    mmt = newTimer(); 

    //opencl context timer
    struct timer *ctxt;
    ctxt = newTimer(); 

    //host side timer 
    struct timer *t;
	t = newTimer();
    timestart(t);

    /*************************************************************************
     *                               setup
     *************************************************************************/
    int bitsincomp = cd->bits;
	int workgroup_size = 1024;//200;
    int stride = workgroup_size;
	int work_per_thread = 1;//400;
    int resultsize;
	int n_workgroups = (bitsincomp / (workgroup_size * work_per_thread)) +
        ((bitsincomp % (workgroup_size * work_per_thread)) == 0 ? 0 : 1);

    //opencl profile timer
    cl_event time_event;
    cl_ulong time_start, time_end;
    
    double initbitsindex_time,
           decodeallbits_time,
           makebigtable_time,
           calcbitsindex_time,
           calcresult_time,
           findmax_time;

    timestart(ctxt);

    if(!platform){
	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*2);
        clGetPlatformIDs(2,          //num_entries
                         platforms,  //platforms
                         NULL);      //num_platforms
	//platform = platforms[0];//intel
	platform = platforms[1];//nvidia
        assert(platform && "No platform");
    }

    if(!device){
        clGetDeviceIDs(platform,    	    //platform_id
#ifdef CPU_BUILD
                       CL_DEVICE_TYPE_CPU,  //device_type
#elif GPU_BUILD
                       CL_DEVICE_TYPE_GPU,  //device_type
#else
                       CL_DEVICE_TYPE_ALL,  //device_type
#endif
                       1,      		    //num_entries
                       &device,             //devices
                       NULL);		    //num_devices,
        assert(device && "No device");
    }

    cl_int error;
    if(!context){
        context = clCreateContext(NULL,   //properties
                                  1,      //num_devices
                                  &device,//devices
                                  NULL,   //pfn_notify
                                  NULL,   //user_data
                                  &error);//errcode_ret
        assert(!error && "No context"); 
    }

    if(!queue){
        queue = clCreateCommandQueue(context,//context
                                     device, //device
                                     CL_QUEUE_PROFILING_ENABLE,//properties
                                     &error);//errcode_ret
        assert(!error && "No queue");
    }
    timestop(ctxt);
    context_time += timerms(ctxt);

    /*************************************************************************
     *                               initbitsindex
     *************************************************************************/
    cl_mem bitsindex_d; //*int
	cl_mem table_d;     //struct HuffNode *
	cl_mem data_d;      //unsigned char* 
	cl_mem bitsteps_d;  //int *
	cl_mem bitdecode_d; //unsigned char *
    cl_mem result_d;    //unsigned char *
    timestart(bct);
    bitsindex_d = clCreateBuffer(context,                 //context
                                 CL_MEM_READ_WRITE,       //flags
                                 sizeof(int)*bitsincomp,  //size
                                 NULL,                    //host_ptr
                                 &error);                 //errcode_ret
    assert(!error && "Failed creating device buffer");
    timestop(bct);
    buffer_creation_time += timerms(bct);

#if defined(BUILD_BINARY_KERNELS)
    bool ok = loadKernelFromSourceAndSaveAsBinary("kernels/initbitsindex.cl",
                                                  "kernels/initbitsindex.bin",
                                                  device,
                                                  context);
    assert(ok && "Failed to build binary kernels [initbitsindex]");
#endif

#if defined(USE_BINARY_KERNELS)
    cl_kernel initbitsindex = getKernelFromBinary("kernels/initbitsindex.bin",
                                                  "initbitsindex",
                                                  device,
                                                  context);
#else //USE_SOURCE
    cl_kernel initbitsindex = getKernelFromSource("kernels/initbitsindex.cl",
                                                  "initbitsindex",
                                                  device,
                                                  context);
#endif
    assert(initbitsindex && "No kernel [initbitsindex]");

    error = clSetKernelArg(initbitsindex,    //kernel
                           0,                //arg_index
                           sizeof(cl_mem),   //arg_size
                           &bitsindex_d);    //arg_value 
    error += clSetKernelArg(initbitsindex,
                            1,
                            sizeof(const int),
                            &bitsincomp);
    error += clSetKernelArg(initbitsindex,
                            2,
                            sizeof(const int),
                            &work_per_thread);
    error += clSetKernelArg(initbitsindex,
                            3,
                            sizeof(const int),
                            &stride);
    assert(!error && "Failed to set kernel arguments");

    size_t global_work_offset[1] = {0};
    size_t global_work_size[1]   = {n_workgroups*workgroup_size};
    size_t local_work_size[1]    = {workgroup_size};
    size_t init_local_work_size[1] = {workgroup_size};

    //gettimeofday(&start_time, NULL); 

    error = clEnqueueNDRangeKernel(queue,             //command_queue
                                   initbitsindex,     //kernel
                                   1,                 //work_dim
                                   global_work_offset,//global_work_offset
                                   global_work_size,  //global_work_size
                                   local_work_size,   //local_work_size
                                   0,                 //number_of_events
                                   NULL,              //wait_list
                                   &time_event);      //event
    assert(!error && "Failed to enqueue kernel [initbitsindex]");
    
    clFinish(queue);

    //gettimeofday(&start_time, NULL);
    //elapsed_time = (stop_time.tv_sec - start_time.tv_sec) +
    //    (stop_time.tv_usec - start_time.tv_usec) / 1000000.0; 

    clGetEventProfilingInfo(time_event,
                            CL_PROFILING_COMMAND_START,
                            sizeof(time_start),
                            &time_start,
                            NULL);
    clGetEventProfilingInfo(time_event,
                            CL_PROFILING_COMMAND_END,
                            sizeof(time_end),
                            &time_end,
                            NULL);
    initbitsindex_time = (time_end - time_start)/1000000.0;
 
#ifdef DEBUG
    printf("Success: Kernel [initbitsindex] took %f milliseconds.\n",
           initbitsindex_time);
#endif
   
#ifdef DEBUG
    /*
    printf("Success: kernel [inibitsindex] took %f milliseconds.\n", elapsed_time);
    int* bitsindex = malloc(sizeof(int)*bitsincomp);
    error = clEnqueueReadBuffer(queue,         //command_queue
                                bitsindex_d,   //buffer
                                CL_TRUE,       //blocking_read
                                0,             //offset
                                bitsincomp*sizeof(int),   //size
                                bitsindex,      //ptr
                                0,             //num_events_in_wait_list
                                NULL,          //event_wait_list
                                NULL);         //event
    assert(!error && "Failed to read buffer [bitsindex_d]");
    bool all_good = true;
    for(unsigned int this_bit = 0; this_bit < bitsincomp; this_bit++){
        if(bitsindex[this_bit] != -1){
            printf("pos %i -> %i\n",this_bit,bitsindex[this_bit]);
            all_good = false;
        }
    }
    assert(all_good);
    free(bitsindex);
    printf("All golden with the bitsindex.\n");
    */
#endif
    /*************************************************************************
     *                               decodeallbits
     *************************************************************************/
    timestart(bct);
    //set up: table, data, bitdecode, bitsteps
    table_d = clCreateBuffer(context,                 //context
                             CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, //flags
                             cd->nodes*sizeof(struct HuffNode),      //size
                             cd->tree,                //host_ptr
                             &error);                 //errcode_ret
    assert(!error && "No device buffer [table_d]");

    int datasize = (
			(bitsincomp % 8) == 0 ? bitsincomp / 8 : (bitsincomp / 8) + 1);
    data_d = clCreateBuffer(context,            //context
                            CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,  //flags
                            datasize*sizeof(unsigned char),          //size
                            cd->data,           //host_ptr
                            &error);            //errcode_ret
    assert(!error && "No device buffer [data_d]");

    bitsteps_d = clCreateBuffer(context,                        //context
                                CL_MEM_READ_WRITE,              //flags
                                25 * bitsincomp * sizeof(int),  //size
                                NULL,                           //host_ptr
                                &error);                        //errcode_ret
    assert(!error && "No device buffer [bitsteps_d]");
	
    result_d = clCreateBuffer(context,                  //context
                              CL_MEM_READ_WRITE,        //flags
                              bitsincomp * sizeof(unsigned char), //size
                              NULL,                     //host_ptr
                              &error);                  //errcode_ret
	assert(!error && "No device buffer [result_d]");

    bitdecode_d = clCreateBuffer(context,                            //context
                                 CL_MEM_READ_WRITE,                  //flags
                                 bitsincomp * sizeof(unsigned char), //size
                                 NULL,                               //host_ptr
                                 &error);                         //errcode_ret
    assert(!error && "No device buffer [bitdecode_d]");
    timestop(bct);
    buffer_creation_time += timerms(bct);

#if defined(BUILD_BINARY_KERNELS)
    ok = loadKernelFromSourceAndSaveAsBinary("kernels/decodeallbits.cl",
                                             "kernels/decodeallbits.bin",
                                             device,
                                             context);
    assert(ok && "Failed to build binary kernels [decodeallbits]");
#endif

#if defined(USE_BINARY_KERNELS)
    cl_kernel decodeallbits = getKernelFromBinary("kernels/decodeallbits.bin",
                                                  "decodeallbits",
                                                  device,
                                                  context);
#else //USE_SOURCE
    cl_kernel decodeallbits = getKernelFromSource("kernels/decodeallbits.cl",
                                                  "decodeallbits",
                                                  device,
                                                  context);
#endif
    assert(decodeallbits && "No kernel [decodeallbits]");

    error = clSetKernelArg(decodeallbits,
                           0,
                           sizeof(const int),
                           &bitsincomp);
    error += clSetKernelArg(decodeallbits,
                            1,
                            sizeof(const int),
                            &work_per_thread);
    error += clSetKernelArg(decodeallbits,
                            2,
                            sizeof(const int),
                            &stride);
    error += clSetKernelArg(decodeallbits,
                            3,
                            sizeof(cl_mem),
                            &table_d);
    error += clSetKernelArg(decodeallbits,
                            4,
                            sizeof(cl_mem),
                            &data_d);
    error += clSetKernelArg(decodeallbits,
                            5,
                            sizeof(cl_mem),
                            &bitdecode_d);
    error += clSetKernelArg(decodeallbits,
                            6,
                            sizeof(cl_mem),
                            &bitsteps_d);
    assert(!error && "Failed to set kernel arguments [decodeallbits]");

    error = clEnqueueNDRangeKernel(queue,             //command_queue
                                   decodeallbits,     //kernel
                                   1,                 //work_dim
                                   global_work_offset,//global_work_offset
                                   global_work_size,  //global_work_size
                                   local_work_size,   //local_work_size
                                   0,                 //number_of_events
                                   NULL,              //wait_list
                                   &time_event);      //event
    assert(!error && "Failed to enqueue kernel [decodeallbits]");

    clFinish(queue);

    clGetEventProfilingInfo(time_event,
                            CL_PROFILING_COMMAND_START,
                            sizeof(time_start),
                            &time_start,
                            NULL);
    clGetEventProfilingInfo(time_event,
                            CL_PROFILING_COMMAND_END,
                            sizeof(time_end),
                            &time_end,
                            NULL);
    decodeallbits_time = (time_end - time_start)/1000000.0;

#ifdef DEBUG
    printf("Success: Kernel [decodeallbits] took %f milliseconds.\n",
           decodeallbits_time);
#endif

#ifdef DEBUG
    /*
    printf("Success: Kernel [decodeallbits] took %f milliseconds.\n", elapsed_time);

    unsigned int max_bit_length = 25 * bitsincomp;
    int* bitsteps = malloc(max_bit_length * sizeof(int));
    error = clEnqueueReadBuffer(queue,         //command_queue
                                bitsteps_d,    //buffer
                                CL_TRUE,       //blocking_read
                                0,             //offset
                                max_bit_length*sizeof(int),   //size
                                bitsteps,      //ptr
                                0,             //num_events_in_wait_list
                                NULL,          //event_wait_list
                                NULL);         //event
    assert(!error && "Failed to read buffer [bitsteps_d]");
    FILE *f = fopen("decodeallbits_from_opencl.txt", "w");
    assert(f != NULL && "Couldn't open file");
    for(unsigned int this_bit = 0; this_bit < bitsincomp; this_bit++){
        fprintf(f,"pos %i -> %i\n",this_bit,bitsteps[this_bit]);
    }
    fclose(f);
    free(bitsteps);
    */
#endif
    /*************************************************************************
     *                               makebigtable
     *************************************************************************/
	int step = 0;
	int powertwo = 1;
	int bitsteoresult;
	cl_mem bitsteoresult_d;//int *

    timestart(bct);
    bitsteoresult_d = clCreateBuffer(context,           //context
                                     CL_MEM_READ_WRITE, //flags
                                     sizeof(int),       //size
                                     NULL,              //host_ptr
                                     &error);           //errcode_ret
    assert(!error && "No device buffer [bitsteoresult_d]");
    timestop(bct);
    buffer_creation_time += timerms(bct);

#if defined(BUILD_BINARY_KERNELS)
    ok = loadKernelFromSourceAndSaveAsBinary("kernels/makebigtable.cl",
                                             "kernels/makebigtable.bin",
                                             device,
                                             context);
    assert(ok && "Failed to build binary kernels [makebigtable]");
#endif

#if defined(USE_BINARY_KERNELS)
    cl_kernel makebigtable = getKernelFromBinary("kernels/makebigtable.bin",
                                                 "makebigtable",
                                                 device,
                                                 context);
#else //USE_SOURCE
    cl_kernel makebigtable = getKernelFromSource("kernels/makebigtable.cl",
                                                 "makebigtable",
                                                 device,
                                                 context);
#endif
    assert(makebigtable && "No kernel [makebigtable]");

    error = clSetKernelArg(makebigtable,
                           0,
                           sizeof(const int),
                           &bitsincomp);
    error += clSetKernelArg(makebigtable,
                            1,
                            sizeof(const int),
                            &work_per_thread);
    error += clSetKernelArg(makebigtable,
                            2,
                            sizeof(const int),
                            &stride);
    error += clSetKernelArg(makebigtable,
                            3,
                            sizeof(cl_mem),
                            &table_d);
    error += clSetKernelArg(makebigtable,
                            4,
                            sizeof(cl_mem),
                            &data_d);
    error += clSetKernelArg(makebigtable,
                            5,
                            sizeof(cl_mem),
                            &bitdecode_d);
    error += clSetKernelArg(makebigtable,
                            6,
                            sizeof(cl_mem),
                            &bitsteps_d);
    error += clSetKernelArg(makebigtable,
                            9,
                            sizeof(cl_mem),
                            &bitsteoresult_d);

    makebigtable_time = 0.0;
	do {
        //update step and powertwo, then invoke the gpu kernel
        error += clSetKernelArg(makebigtable,
                                7,
                                sizeof(const int),
                                &step);
        error += clSetKernelArg(makebigtable,
                                8,
                                sizeof(const int),
                                &powertwo);
        assert(!error && "Failed to set kernel arguments [makebigtable]");

        error = clEnqueueNDRangeKernel(queue,             //command_queue
                                       makebigtable,      //kernel
                                       1,                 //work_dim
                                       global_work_offset,//global_work_offset
                                       global_work_size,  //global_work_size
                                       local_work_size,   //local_work_size
                                       0,                 //number_of_events
                                       NULL,              //wait_list
                                       &time_event);      //event
        assert(!error && "Failed to enqueue kernel [makebigtable]");

        clFinish(queue);
        clGetEventProfilingInfo(time_event,
                                CL_PROFILING_COMMAND_START,
                                sizeof(time_start),
                                &time_start,
                                NULL);
        clGetEventProfilingInfo(time_event,
                                CL_PROFILING_COMMAND_END,
                                sizeof(time_end),
                                &time_end,
                                NULL);
        makebigtable_time += (time_end - time_start)/1000000.0;

        //collect bisteoresult from gpu's intermediate result
        timestart(mmt);
        error = clEnqueueReadBuffer(queue,              //command_queue
                                    bitsteoresult_d,    //buffer
                                    CL_TRUE,            //blocking_read
                                    0,                  //offset
                                    sizeof(int),        //size
                                    &bitsteoresult,     //ptr
                                    0,              //num_events_in_wait_list
                                    NULL,           //event_wait_list
                                    NULL);          //event
        assert(!error && "Failed to read buffer [bitsteoresult_d]");
        timestop(mmt);
        memory_movement_time += timerms(mmt);

		step++;
		powertwo = powertwo << 1;
	} while (bitsteoresult != -1);

#ifdef DEBUG
    printf("Success: Kernel [makebigtable] took %f milliseconds.\n",
           makebigtable_time);
#endif
    /*************************************************************************
     *                               calcbitsindex
     *************************************************************************/
	int zerovalue = 0;
	powertwo = powertwo >> 1;
    //zero the gpu's copy of bitsindex_d 
    timestart(mmt);
    error = clEnqueueWriteBuffer(queue,          //command_queue
                                 bitsindex_d,    //buffer
                                 CL_TRUE,        //blocking_read
                                 0,              //offset
                                 sizeof(int),    //size
                                 &zerovalue,     //ptr
                                 0,              //num_events_in_wait_list
                                 NULL,           //event_wait_list
                                 NULL);          //event
    assert(!error && "Failed to write buffer [bitsindex_d]");
    timestop(mmt);
    memory_movement_time += timerms(mmt);

#if defined(BUILD_BINARY_KERNELS)
    ok = loadKernelFromSourceAndSaveAsBinary("kernels/calcbitsindex.cl",
                                             "kernels/calcbitsindex.bin",
                                             device,
                                             context);
    assert(ok && "Failed to build binary kernels [calcbitsindex]");
#endif

#if defined(USE_BINARY_KERNELS)
    cl_kernel calcbitsindex = getKernelFromBinary("kernels/calcbitsindex.bin",
                                                  "calcbitsindex",
                                                  device,
                                                  context);
#else //USE_SOURCE
    cl_kernel calcbitsindex = getKernelFromSource("kernels/calcbitsindex.cl",
                                                  "calcbitsindex",
                                                  device,
                                                  context);
#endif
    assert(calcbitsindex && "No kernel [calcbitsindex]");

    error = clSetKernelArg(calcbitsindex,
                           0,
                           sizeof(const int),
                           &bitsincomp);
    error += clSetKernelArg(calcbitsindex,
                            1,
                            sizeof(const int),
                            &work_per_thread);
    error += clSetKernelArg(calcbitsindex,
                            2,
                            sizeof(const int),
                            &stride);
    error += clSetKernelArg(calcbitsindex,
                            3,
                            sizeof(cl_mem),
                            &bitsindex_d);
    error += clSetKernelArg(calcbitsindex,
                            4,
                            sizeof(cl_mem),
                            &bitsteps_d);

    calcbitsindex_time = 0.0;
	while (step > 0) {
        //update step and powertwo, then invoke the gpu kernel
        error += clSetKernelArg(calcbitsindex,
                                5,
                                sizeof(const int),
                                &step);
        error += clSetKernelArg(calcbitsindex,
                                6,
                                sizeof(const int),
                                &powertwo);
        assert(!error && "Failed to set kernel arguments [calcbitsindex]");

        error = clEnqueueNDRangeKernel(queue,             //command_queue
                                       calcbitsindex,     //kernel
                                       1,                 //work_dim
                                       global_work_offset,//global_work_offset
                                       global_work_size,  //global_work_size
                                       local_work_size,   //local_work_size
                                       0,                 //number_of_events
                                       NULL,              //wait_list
                                       &time_event);      //event
        assert(!error && "Failed to enqueue kernel [calcbitsindex]");

        clFinish(queue);
        clGetEventProfilingInfo(time_event,
                                CL_PROFILING_COMMAND_START,
                                sizeof(time_start),
                                &time_start,
                                NULL);
        clGetEventProfilingInfo(time_event,
                                CL_PROFILING_COMMAND_END,
                                sizeof(time_end),
                                &time_end,
                                NULL);
        calcbitsindex_time += (time_end - time_start)/1000000.0;

        step--;
		powertwo = powertwo >> 1;
	}

#ifdef DEBUG
    printf("Success: Kernel [calcbitsindex] took %f milliseconds.\n",
           calcbitsindex_time);
#endif
    /*************************************************************************
     *                               calcresult
     *************************************************************************/
#if defined(BUILD_BINARY_KERNELS)
    ok = loadKernelFromSourceAndSaveAsBinary("kernels/calcresult.cl",
                                             "kernels/calcresult.bin",
                                             device,
                                             context);
    assert(ok && "Failed to build binary kernels [calcresult]");
#endif

#if defined(USE_BINARY_KERNELS)
    cl_kernel calcresult = getKernelFromBinary("kernels/calcresult.bin",
                                               "calcresult",
                                               device,
                                               context);
#else //USE_SOURCE
    cl_kernel calcresult = getKernelFromSource("kernels/calcresult.cl",
                                               "calcresult",
                                               device,
                                               context);
#endif
    assert(calcresult && "No kernel [calcresult]");

    error = clSetKernelArg(calcresult,
                           0,
                           sizeof(const int),
                           &bitsincomp);
    error += clSetKernelArg(calcresult,
                            1,
                            sizeof(const int),
                            &work_per_thread);
    error += clSetKernelArg(calcresult,
                            2,
                            sizeof(const int),
                            &stride);
    error += clSetKernelArg(calcresult,
                            3,
                            sizeof(cl_mem),
                            &bitsindex_d);
    error += clSetKernelArg(calcresult,
                            4,
                            sizeof(cl_mem),
                            &bitdecode_d);
    error += clSetKernelArg(calcresult,
                            5,
                            sizeof(cl_mem),
                            &result_d);
    assert(!error && "Failed to set kernel arguments [calcresult]");

    error = clEnqueueNDRangeKernel(queue,             //command_queue
                                   calcresult,        //kernel
                                   1,                 //work_dim
                                   global_work_offset,//global_work_offset
                                   global_work_size,  //global_work_size
                                   local_work_size,   //local_work_size
                                   0,                 //number_of_events
                                   NULL,              //wait_list
                                   &time_event);      //event
    assert(!error && "Failed to enqueue kernel [calcresult]");

    clFinish(queue);
    clGetEventProfilingInfo(time_event,
                            CL_PROFILING_COMMAND_START,
                            sizeof(time_start),
                            &time_start,
                            NULL);
    clGetEventProfilingInfo(time_event,
                            CL_PROFILING_COMMAND_END,
                            sizeof(time_end),
                            &time_end,
                            NULL);
    calcresult_time = (time_end - time_start)/1000000.0;

#ifdef DEBUG
    printf("Success: Kernel [calcresult] took %f milliseconds.\n", calcresult_time);
#endif
    /*************************************************************************
     *                               findmax
     *************************************************************************/
    int maxvalue;
#if defined(BUILD_BINARY_KERNELS)
    ok = loadKernelFromSourceAndSaveAsBinary("kernels/findmax.cl",
                                             "kernels/findmax.bin",
                                             device,
                                             context);
    assert(ok && "Failed to build binary kernels [findmax]");
#endif

#if defined(USE_BINARY_KERNELS)
    cl_kernel findmax = getKernelFromBinary("kernels/findmax.bin",
                                            "findmax",
                                            device,
                                            context);
#else //USE_SOURCE
    cl_kernel findmax = getKernelFromSource("kernels/findmax.cl",
                                            "findmax",
                                            device,
                                            context);
#endif
    assert(findmax && "No kernel [findmax]");

    error = clSetKernelArg(findmax,
                           0,
                           sizeof(const int),
                           &bitsincomp);
    error += clSetKernelArg(findmax,
                            1,
                            sizeof(cl_mem),
                            &bitsindex_d);
    assert(!error /*"Failed to set kernel arguments [findmax]"*/);

    size_t one_global_size[1] = {1};
    size_t one_local_size[1]  = {1};

    error = clEnqueueNDRangeKernel(queue,               //command_queue
                                   findmax,             //kernel
                                   1,                   //work_dim
                                   global_work_offset,  //global_work_offset
                                   one_global_size,     //global_work_size
                                   one_local_size,      //local_work_size
                                   0,                   //number_of_events
                                   NULL,                //wait_list
                                   &time_event);        //event
    assert(!error && "Failed to enqueue kernel [findmax]");

    clFinish(queue);
    clGetEventProfilingInfo(time_event,
                            CL_PROFILING_COMMAND_START,
                            sizeof(time_start),
                            &time_start,
                            NULL);
    clGetEventProfilingInfo(time_event,
                            CL_PROFILING_COMMAND_END,
                            sizeof(time_end),
                            &time_end,
                            NULL);
    findmax_time = (time_end - time_start)/1000000.0;

#ifdef DEBUG
    printf("Success: Kernel [findmax] took %f milliseconds.\n", findmax_time);
#endif
    //collect bisteoresult from gpu's intermediate result
    timestart(mmt);
    error = clEnqueueReadBuffer(queue,          //command_queue
                                bitsindex_d,    //buffer
                                CL_TRUE,        //blocking_read
                                0,              //offset
                                sizeof(int),    //size
                                &maxvalue,      //ptr
                                0,              //num_events_in_wait_list
                                NULL,           //event_wait_list
                                NULL);          //event
    assert(!error /*"Failed to read buffer [bitsindex_d]"*/);
    
    /*************************************************************************
     *                               cleanup
     *************************************************************************/
	// obtain result from GPU
    resultsize = maxvalue + 1;
    error = clEnqueueReadBuffer(queue,          //command_queue
                                result_d,       //buffer
                                CL_TRUE,        //blocking_read
                                0,              //offset
                                resultsize*sizeof(unsigned char),   //size
                                uncompressed->data,                 //ptr
                                0,              //num_events_in_wait_list
                                NULL,           //event_wait_list
                                NULL);          //event
    assert(!error /*"Failed to read buffer [result_d]"*/);
    timestop(mmt);
    memory_movement_time += timerms(mmt);

    clReleaseKernel(findmax);
    clReleaseKernel(calcresult);
    clReleaseKernel(calcbitsindex);
    clReleaseKernel(makebigtable);
    clReleaseKernel(decodeallbits);
    clReleaseKernel(initbitsindex);
    clReleaseMemObject(bitsteoresult_d);
    clReleaseMemObject(bitdecode_d);
    clReleaseMemObject(result_d);
    clReleaseMemObject(bitsteps_d);
    clReleaseMemObject(data_d);
    clReleaseMemObject(table_d);
    clReleaseMemObject(bitsindex_d);

    timestop(t);
    double host_time = timerms(t);

    double total_time = initbitsindex_time + decodeallbits_time +
        makebigtable_time + calcbitsindex_time + calcresult_time + findmax_time;

    //printf("Total time took : %f (ms)\n", host_time);
    //printf("Actual computation took : %f (ms)\n", total_time);
    //printf("Context setup took : %f (ms)\n", context_time);
    //printf("Kernel comilation time took : %f (ms)\n", build_time);
    //printf("Buffer creation time took : %f (ms)\n", buffer_creation_time);
    //printf("Memory movement time took : %f (ms)\n", memory_movement_time);
    //printf("Unaccounted host time took : %f (ms)\n", host_time-total_time);
    
}

void end (void){
    clReleaseDevice(device);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

