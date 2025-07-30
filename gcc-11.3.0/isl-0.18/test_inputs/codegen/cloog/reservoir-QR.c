if (N >= 1) {
  S1(0);
  if (N == 1) {
    for (int c1 = 0; c1 < M; c1 += 1)
      S2(0, c1);
    S3(0);
    for (int c1 = 0; c1 < M; c1 += 1)
      S4(0, c1);
    S10(0);
    S5(0);
  } else {
    for (int c1 = 0; c1 < M; c1 += 1)
      S2(0, c1);
    S3(0);
    for (int c1 = 0; c1 < M; c1 += 1)
      S4(0, c1);
    S10(0);
    S1(1);
    S5(0);
  }
  for (int c0 = 2; c0 < N; c0 += 1) {
    for (int c1 = c0 - 1; c1 < N; c1 += 1) {
      S6(c0 - 2, c1);
      for (int c2 = c0 - 2; c2 < M; c2 += 1)
        S7(c0 - 2, c1, c2);
      S8(c0 - 2, c1);
      for (int c2 = c0 - 2; c2 < M; c2 += 1)
        S9(c0 - 2, c1, c2);
    }
    for (int c1 = c0 - 1; c1 < M; c1 += 1)
      S2(c0 - 1, c1);
    S3(c0 - 1);
    for (int c1 = c0 - 1; c1 < M; c1 += 1)
      S4(c0 - 1, c1);
    S10(c0 - 1);
    S1(c0);
    S5(c0 - 1);
  }
  if (N >= 2) {
    S6(N - 2, N - 1);
    for (int c2 = N - 2; c2 < M; c2 += 1)
      S7(N - 2, N - 1, c2);
    S8(N - 2, N - 1);
    for (int c2 = N - 2; c2 < M; c2 += 1)
      S9(N - 2, N - 1, c2);
    for (int c1 = N - 1; c1 < M; c1 += 1)
      S2(N - 1, c1);
    S3(N - 1);
    for (int c1 = N - 1; c1 < M; c1 += 1)
      S4(N - 1, c1);
    S10(N - 1);
    S5(N - 1);
  }
}
