/*
 * timing.c
 *
 *  Created on: 03/12/2015
 *      Author: ericm
 */

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include "timing.h"
#include "time.h"

struct timer * newTimer() {
	struct timer *res;
	res = (struct timer *) malloc(sizeof(struct timer));
	if (!res) {fprintf(stderr, "malloc timer"); exit(-1);}
	return res;
}

void freeTimer(struct timer *t) {
	free(t);
}

void timestart(struct timer *t) {
    get_time(&(t->tv1));
	//clock_gettime(CLOCK_MONOTONIC_RAW,&(t->tv1));
	//gettimeofday(&(t->tv1), NULL);
}

void timestop(struct timer *t) {
    get_time(&(t->tv2));
	//clock_gettime(CLOCK_MONOTONIC_RAW,&(t->tv2));
	//gettimeofday(&(t->tv2), NULL);
}

void timereport(struct timer *t) {
	printf("%f\n", timerseconds(t));
}

void timereportns(struct timer *t) {
	printf("%ld\n", timerns(t));
}


void reportresolution() {
	struct timespec ts;
	get_time_res(&ts);
	printf("CLOCK_MONOTONIC_RAW resoltion %ld sec %ld ns\n",ts.tv_sec, ts.tv_nsec);
}

double timerseconds(struct timer *t) {
	return (double) (t->tv2.tv_nsec - t->tv1.tv_nsec) / 1000000000.0
			+ (double) (t->tv2.tv_sec - t->tv1.tv_sec);
}

double timerms(struct timer *t) {
	return (double) (t->tv2.tv_nsec - t->tv1.tv_nsec) / 1000000.0
			+ (double) (t->tv2.tv_sec - t->tv1.tv_sec) * 1000.0;
}

long timerns(struct timer *t) {
	return  (t->tv2.tv_nsec - t->tv1.tv_nsec) +
			+ ((long) (t->tv2.tv_sec - t->tv1.tv_sec)) * 1000000000L;
}

