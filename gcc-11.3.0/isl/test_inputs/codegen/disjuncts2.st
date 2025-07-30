# Check that the loop is generated only once with an outer disjunctive condition
domain: "[N, Q, P] -> { S[i0] : 0 <= i0 < N and (P > Q or P < Q) }"
child:
  schedule: "[N, Q, P] -> [{ S[i0] -> [(i0)] }]"
