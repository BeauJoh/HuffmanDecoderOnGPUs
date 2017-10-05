/*
 * linapproach.c
 *
 *  Created on: 22/12/2015
 *      Author: ericm
 */

#include "error.h"
#include <stdlib.h>
#include<stdio.h>
#include <assert.h>

#include"linapproach.h"
#include"huffdata.h"

void findroots(int down, int jumpbits, struct HuffNode *tree, int root,
		int *roots, int *rootscount) {
	if (tree[root].izero == -1) {
		return;
	} else if (down == 0) {
		roots[*rootscount] = root;
		*rootscount += 1;
		findroots(jumpbits, jumpbits, tree, root, roots, rootscount);
	} else {
		findroots(down - 1, jumpbits, tree, tree[root].izero, roots,
				rootscount);
		findroots(down - 1, jumpbits, tree, tree[root].ione, roots, rootscount);

	}
}

void findteleroots(int level, int offset, int down, struct HuffNode *tree,
		int root, int *roots, int *rootlevel, int *rootscount) {
	if (tree[root].izero == -1) {
		return;
	} else if (down == 0) {
		return;
	} else {
		roots[offset + *rootscount] = root;
		rootlevel[*rootscount] = level;
		(*rootscount)++;
		findteleroots(level + 1, offset, down - 1, tree, tree[root].izero,
				roots, rootlevel, rootscount);
		findteleroots(level + 1, offset, down - 1, tree, tree[root].ione, roots,
				rootlevel, rootscount);
	}
}

void traverseLinTree(struct HuffNode *tree, struct sElement8 *se, int jumpbits,
		int code, int subtreeroot, int *invertroot, int istele, int telelevel) {
	int nextbit;
	int pos = 0;
	int tablepos = subtreeroot;
	int resultpos = 0;
	int backbits = 0;
	int lastback = 0;
	while (pos < jumpbits) {
		nextbit = (code >> pos) & 1;
		tablepos = (nextbit ? tree[tablepos].ione : tree[tablepos].izero);
		pos++;
		if (istele && ((telelevel + pos) == jumpbits)) {
			backbits = jumpbits - pos;
			lastback = tablepos;
		}

		if (tree[tablepos].izero == -1) { // its a leaf
			se->syms[resultpos] = tree[tablepos].sym;
			resultpos++;
			tablepos = 0;
		}

	}
	if (resultpos > 0) {
		se->numSym = resultpos;
		se->l = invertroot[tablepos];
	} else {
		se->numSym = 0;
		if (istele) {
			se->l = backbits;
			se->val = invertroot[lastback];
		} else {
			se->l = 0;
			se->val = invertroot[tablepos];
		}
	}

}

void maketableLin(int table, int atablesize, struct sElement8 * stable,
		int subtreeroot, int *invertroots, struct HuffNode *tree, int jumpbits,
		int istele, int telelevel) {
	int j;
	struct sElement8 *se;
#ifdef DEBUG
	printf("maketableLin table %d subtreeroot %d istele %d telelevel %d\n", table, subtreeroot, istele, telelevel);
#endif
	for (j = 0; j < atablesize; j++) {
		se = &(stable[table * atablesize + j]);
		se->numSym = 0;
		traverseLinTree(tree, se, jumpbits, j, subtreeroot, invertroots, istele,
				telelevel);

	}

}

int maskLin[] = { 0x0000, 0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F,
		0x007F, 0x00FF, 0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF };

void linApproach(struct CompressedData *cd,
		struct UnCompressedData *uncompressed, void *paramdata) {
	int   jumpbits, atablesize, maxnumbertables,
			nonteleroots, teleroots;
	// int treesize, mindepth, height;

//	mindepth = tableMinDepth(cd->tree, 0);
//	height = tableHeight(cd->tree, 0);
//	treesize = treeSize(cd->tree, 0);

	jumpbits = *((int *) paramdata);
	atablesize = 1 << jumpbits;
	nonteleroots = tableNumGroups(cd->tree, jumpbits, 0);
	teleroots = telescoped(cd->tree, jumpbits, 0);
	maxnumbertables = nonteleroots + teleroots;

	struct sElement8 *stable;
	stable = (struct sElement8 *) malloc(
			sizeof(struct sElement8) * maxnumbertables * atablesize);

	int *roots;
	int *rootlevels;
	roots = (int *) malloc(maxnumbertables * sizeof(int));
	rootlevels = (int *) malloc(teleroots * sizeof(int));
	int rootscount, telerootscount;

	int *invertroots;
	invertroots = (int *) malloc(cd->nodes * sizeof(int));

	rootscount = 1;
	roots[0] = 0;
	findroots(jumpbits, jumpbits, cd->tree, 0, roots, &rootscount);

	assert(rootscount == nonteleroots);

	telerootscount = 0;
	if (cd->tree[0].izero != -1) {
		findteleroots(1, nonteleroots, jumpbits - 1, cd->tree,
				cd->tree[0].izero, roots, rootlevels, &telerootscount);
		findteleroots(1, nonteleroots, jumpbits - 1, cd->tree, cd->tree[0].ione,
				roots, rootlevels, &telerootscount);
	}

	assert(telerootscount == teleroots);

	int i;
	for (i = 0; i < maxnumbertables; i++)
		invertroots[roots[i]] = i * atablesize;

#ifdef DEBUG
	printf("rootscount %d nontele %d tele %d\n", maxnumbertables, nonteleroots,
			teleroots);
	for (i = 0; i < maxnumbertables; i++)
	printf("%d ", roots[i]);
	printf("\n");

	for (i = 0; i < cd->nodes; i++)
	printf("%d ", invertroots[i]);
	printf("\n");
#endif

	for (i = 0; i < nonteleroots; i++) {
		maketableLin(i, atablesize, stable, roots[i], invertroots, cd->tree,
				jumpbits, 0, 0);
	}

	for (i = 0; i < teleroots; i++) {
		maketableLin(i + nonteleroots, atablesize, stable,
				roots[i + nonteleroots], invertroots, cd->tree, jumpbits, 1,
				rootlevels[i]);
	}

/*
#ifdef DEBUG
	for (i = 0; i < maxnumbertables * atablesize; i++) {
		printf("%d: ", i);
		if (stable[i].numSym > 0) {
			printf("l: %d  ", stable[i].l);
			for (j = 0; j < stable[i].numSym; j++)
			printf("'%c'", stable[i].syms[j]);
			printf("\n");
		} else {
			printf("l: %d s: %d\n", stable[i].l, stable[i].val);
		}
	}
#endif
*/
	// lets decode!!
	int index;
	int code_ptr = 0;
	int array_ptr = 0;
	int resultpos = 0;

	if (jumpbits < 8) {

		while (code_ptr < cd->bits) {

			index = ((cd->data[code_ptr >> 3]
					| ((cd->data[(code_ptr >> 3) + 1]) << 8)) >> (code_ptr & 7))
					& maskLin[jumpbits];

			array_ptr += index;

			if (stable[array_ptr].numSym == 0) {
#ifdef DEBUG
				printf("ap:%d  syms: %d subbits:%d nexttab:%d\n",array_ptr, stable[array_ptr].numSym, stable[array_ptr].l, stable[array_ptr].val);
#endif
				code_ptr += (jumpbits - stable[array_ptr].l);
				array_ptr = stable[array_ptr].val;
			} else {
#ifdef DEBUG
				printf("ap:%d  syms: %d l:%d respos:%d \"",array_ptr, stable[array_ptr].numSym, stable[array_ptr].l, resultpos);
#endif
				for (i = 0; i < stable[array_ptr].numSym; i++) {
					uncompressed->data[resultpos] = stable[array_ptr].syms[i];
#ifdef DEBUG
					printf("%c", stable[array_ptr].syms[i]);
#endif
					resultpos++;
				}
#ifdef DEBUG
				printf("\"\n");
#endif
				array_ptr = stable[array_ptr].l;
				code_ptr += jumpbits;
			}

		}
	} else { // jumpbits > 8

		while (code_ptr < cd->bits) {

			index =
					((cd->data[code_ptr >> 3]
							| ((cd->data[(code_ptr >> 3) + 1]) << 8)
							| ((cd->data[(code_ptr >> 3) + 2]) << 16))
							>> (code_ptr & 7)) & maskLin[jumpbits];

			array_ptr += index;

			if (stable[array_ptr].numSym == 0) {
#ifdef DEBUG
				printf("ap:%d  syms: %d subbits:%d nexttab:%d\n",array_ptr, stable[array_ptr].numSym, stable[array_ptr].l, stable[array_ptr].val);
#endif
				code_ptr += (jumpbits - stable[array_ptr].l);
				array_ptr = stable[array_ptr].val;
			} else {
#ifdef DEBUG
				printf("ap:%d  syms: %d l:%d respos:%d \"",array_ptr, stable[array_ptr].numSym, stable[array_ptr].l, resultpos);
#endif
				for (i = 0; i < stable[array_ptr].numSym; i++) {
					uncompressed->data[resultpos] = stable[array_ptr].syms[i];
#ifdef DEBUG
					printf("%c", stable[array_ptr].syms[i]);
#endif
					resultpos++;
				}
#ifdef DEBUG
				printf("\"\n");
#endif
				array_ptr = stable[array_ptr].l;
				code_ptr += jumpbits;
			}

		}

	}
	free(rootlevels);
	free(invertroots);
	free(roots);

	free(stable);
}
