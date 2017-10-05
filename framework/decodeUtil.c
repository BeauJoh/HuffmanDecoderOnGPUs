/*
 * decodeUtil.c
 *
 *  Created on: 03/12/2015
 *      Author: ericm
 */

#include "decodeUtil.h"
#include "huffdata.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include "timing.h"

struct decoder *newDecoder(void (*decoder_function), void *paramdata,
		char *name) {
	struct decoder *res;
	res = (struct decoder *) malloc(sizeof(struct decoder));
	res->decoder_function = decoder_function;
	res->paramdata = paramdata;
	res->name = name;
	return res;
}

void freeDecoder(struct decoder *d) {
	free(d);
}

double evaluate(struct decoder *d, struct TestData *td, int withcheck) {

	struct UnCompressedData *ucddecoded;
	struct timer *t;
	t = newTimer();

	//infoCompressedData(cd);
	ucddecoded = newUnCompressedData(td->cd->uncompressedsize);
	clearUnCompressedData(ucddecoded);
	int i;
//for (i=0;i<3;i++)printf("%d %d\n", ucddecoded->data[i+100], td->ucd->data[i+100]);
	timestart(t);
	d->decoder_function(td->cd, ucddecoded, d->paramdata);
	timestop(t);

	double mintime = timerseconds(t);

	if (withcheck) {
		if (compareUnCompressedData(ucddecoded, td->ucd) != 0) {
			fprintf(stderr, "problem with : %s\n", d->name);
			err(1, "decode problem");
		}
	}

	for (i = 0; i < REPEATS; i++) {
		clearUnCompressedData(ucddecoded);

		timestart(t);
		d->decoder_function(td->cd, ucddecoded, d->paramdata);
		timestop(t);
	//	timereportns(t);
		if (timerseconds(t) < mintime) {
			mintime = timerseconds(t);
		}
	}
    freeTimer(t);
	freeUnCompressedData(ucddecoded);

	return mintime;

}
