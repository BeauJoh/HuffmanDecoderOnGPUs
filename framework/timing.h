/*
 * timing.h
 *
 *  Created on: 03/12/2015
 *      Author: ericm
 */

#ifndef TIMING_H_
#define TIMING_H_

#include <sys/time.h>
#include <stdio.h>
#include <time.h>

struct timer {
//    struct timeval tv1, tv2;
	 struct timespec  tv1, tv2;
};

struct timer *newTimer();
void freeTimer(struct timer *t);

void timestart(struct timer *t);
void timestop(struct timer *t);
void timereport(struct timer *t);
void timereportns(struct timer *t);
double timerseconds(struct timer *t);
void reportresolution();
double timerms(struct timer *t);
long timerns(struct timer *t);

#endif /* TIMING_H_ */
