#include <time.h>
#include <sys/time.h>
#include <stdio.h>

#ifdef __APPLE__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

void get_time(struct timespec *ts) {
#ifdef __APPLE__ // OS X does not have clock_gettime, use clock_get_time
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    ts->tv_sec = mts.tv_sec;
    ts->tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_MONOTONIC_RAW, ts);
#endif
}

void get_time_res(struct timespec *ts){
#ifdef __APPLE__
    clock_serv_t cclock;
    host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);
    int nano_secs;
    mach_msg_type_number_t count;
    clock_get_attributes(cclock,CLOCK_GET_TIME_RES,(clock_attr_t)&nano_secs,&count);
    mach_port_deallocate(mach_task_self(), cclock);
    ts->tv_nsec = nano_secs;
#else
    clock_getres(CLOCK_MONOTONIC_RAW, ts);
#endif
}
