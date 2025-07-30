{
  if (N >= 2)
    for (int c0 = 1; c0 < O; c0 += 1) {
      for (int c3 = 1; c3 <= M; c3 += 1)
        S1(c0, 1, c3);
      for (int c3 = 1; c3 < M; c3 += 1) {
        S6(c0, 1, c3);
        S7(c0, 1, c3);
      }
      if (N >= 3) {
        for (int c3 = 1; c3 <= M; c3 += 1)
          S3(c0, 1, c3);
        for (int c3 = 1; c3 <= M; c3 += 1)
          S1(c0, 2, c3);
        for (int c3 = 1; c3 < M; c3 += 1) {
          S6(c0, 2, c3);
          S7(c0, 2, c3);
        }
        for (int c3 = 1; c3 < M; c3 += 1)
          S11(c0, 1, c3);
      } else {
        for (int c3 = 1; c3 <= M; c3 += 1)
          S3(c0, 1, c3);
        for (int c3 = 1; c3 < M; c3 += 1)
          S11(c0, 1, c3);
      }
      for (int c1 = 3; c1 < 2 * N - 4; c1 += 2) {
        for (int c3 = 1; c3 < M; c3 += 1)
          S10(c0, (c1 - 1) / 2, c3);
        for (int c3 = 1; c3 <= M; c3 += 1)
          S3(c0, (c1 + 1) / 2, c3);
        for (int c3 = 1; c3 <= M; c3 += 1)
          S1(c0, (c1 + 3) / 2, c3);
        for (int c3 = 1; c3 < M; c3 += 1) {
          S6(c0, (c1 + 3) / 2, c3);
          S7(c0, (c1 + 3) / 2, c3);
        }
        for (int c3 = 1; c3 < M; c3 += 1)
          S11(c0, (c1 + 1) / 2, c3);
      }
      if (N >= 3) {
        for (int c3 = 1; c3 < M; c3 += 1)
          S10(c0, N - 2, c3);
        for (int c3 = 1; c3 <= M; c3 += 1)
          S3(c0, N - 1, c3);
        for (int c3 = 1; c3 < M; c3 += 1)
          S11(c0, N - 1, c3);
      }
      for (int c3 = 1; c3 < M; c3 += 1)
        S10(c0, N - 1, c3);
    }
  for (int c0 = 1; c0 < O; c0 += 1)
    for (int c1 = 1; c1 < N; c1 += 1) {
      for (int c3 = 1; c3 <= M; c3 += 1)
        S2(c0, c1, c3);
      for (int c3 = 1; c3 < M; c3 += 1)
        S8(c0, c1, c3);
      for (int c3 = 1; c3 < M; c3 += 1)
        S9(c0, c1, c3);
    }
  for (int c0 = 1; c0 < O; c0 += 1)
    for (int c1 = 1; c1 < N; c1 += 1)
      for (int c2 = 1; c2 < M; c2 += 1)
        S4(c0, c1, c2);
  for (int c0 = 1; c0 < O; c0 += 1)
    for (int c1 = 1; c1 < N; c1 += 1)
      for (int c2 = 1; c2 < M; c2 += 1)
        S5(c0, c1, c2);
  for (int c0 = R; c0 < O; c0 += 1)
    for (int c1 = Q; c1 < N; c1 += 1)
      for (int c2 = P; c2 < M; c2 += 1)
        S12(c0, c1, c2);
  for (int c0 = R; c0 < O; c0 += 1)
    for (int c1 = Q; c1 < N; c1 += 1)
      for (int c2 = 1; c2 < M; c2 += 1)
        S13(c0, c1, c2);
  for (int c0 = R; c0 < O; c0 += 1)
    for (int c1 = 1; c1 < N; c1 += 1)
      for (int c2 = P; c2 < M; c2 += 1)
        S14(c0, c1, c2);
  for (int c0 = R; c0 < O; c0 += 1)
    for (int c1 = 1; c1 < N; c1 += 1)
      for (int c2 = 1; c2 < M; c2 += 1)
        S15(c0, c1, c2);
}
