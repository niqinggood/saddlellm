# Methodology open decisions

These decisions must be frozen before the confirmatory experiments and then
reflected in `04_methodology.tex`.

1. Final operation taxonomy and typed edge vocabulary for RTGs.
2. Number and decoding configuration of base-policy traces used by the graph
   extractor, source learnability estimator, and target headroom estimator.
3. Graph extractor model, prompt, JSON schema, and human-audit protocol.
4. Motif radius/size, graph distance, kernel temperature, and sparse-kernel
   construction.
5. Whether target answer entropy alone is sufficient for label-free headroom;
   pre-register alternatives before looking at held-out transfer results.
6. Meta-training/meta-development/held-out-domain split used to calibrate DRC
   and the transfer predictor.
7. Transfer-predictor family and the exact prespecified control variables.
8. Beta prior, sampling budget, solvability threshold, confidence threshold,
   and sharpening margin used by Capability Flow.
9. Token-cost definition, diversity weight, and stopping rule for DRC-Select.
10. Exact information-access policy for the unlabeled target development set
    and the locked target test set.
11. Proof or qualified wording for any submodularity/approximation claim about
    the DRC-Select objective.
12. Terminology consistency: use "empirical support expansion" unless stronger
    evidence justifies a capability-acquisition claim.
