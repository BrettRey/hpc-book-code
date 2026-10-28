/* Parameters generated from a floating-point version of the map; Q24.
 * LEARNER0: learners' starting histogram, clamped N(0.90, sd 0.1414), sums exactly to ONE.
 * W_*: intent weights (one item, small exact number, vague many, existence, mass quantity),
 *      a fixed mix of the two semantic bands present in the agent model.
 * SEM_*: the noun's meaning-side value that coupling pulls toward. */
static const int64_t LEARNER0[NB] = {0, 0, 0, 0, 0, 1, 2, 4, 9, 21, 47, 102, 216, 443, 879, 1691, 3154, 5702, 9993, 16976, 27953, 44614, 69021, 103504, 150450, 211980, 289507, 383254, 491788, 611692, 737482, 861854, 976291, 1071985, 1140936, 1177058, 1177058, 1140936, 1071985, 4998628};
static const int64_t W_PRE05[5] = {4496294, 3523215, 4496294, 2852127, 1409286};
static const int64_t W_PRE0[5] = {5200937, 3774874, 4194304, 2600468, 1006633};
#define SEM_PRE05 13891535
#define SEM_PRE0 14327742
