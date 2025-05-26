#ifndef SpMM_OPT_H
#define SpMM_OPT_H
#include "spmm_base.h"

#define WARP_SIZE 32
#define SLICE_SIZE 256

#define NUMV_ARXIV 169343
#define NUMV_COLLAB 235868
#define NUMV_CITATION 2927963
#define NUMV_DDI 4267
#define NUMV_PROTEIN 132534
#define NUMV_PPA 576289
#define NUMV_REDDIT_DGL 232965
#define NUMV_PRODUCTS 2449029
#define NUMV_YOUTUBE 1138499
#define NUMV_AMAZON_COGDL 1569960
#define NUMV_YELP 716847
#define NUMV_WIKIKG2 2500604
#define NUMV_AM 881680

class SpMMOpt : public SpMM
{
public:
    SpMMOpt(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in) : SpMM(dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in) {}
    SpMMOpt(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}
    ~SpMMOpt() {
        if (target) checkCudaErrors(cudaFree(target));
        if (ptr_scheduled) checkCudaErrors(cudaFree(ptr_scheduled));
    }
     
    virtual void preprocess(float *vin, float *vout);

    virtual void run(float *vin, float *vout);

    void edgesort();

    void neighbor_grouping(int neighbor_num);

private:
    int num_target;
    int *target, *ptr_scheduled;
    bool slice = false;
    int *d_rows = nullptr, *d_starts = nullptr, *d_ends = nullptr;
    int num_s = 0;
    int speedup = 1;
};
#endif