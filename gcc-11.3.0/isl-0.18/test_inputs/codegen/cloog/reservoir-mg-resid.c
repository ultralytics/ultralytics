for (int c0 = 2; c0 < O; c0 += 1)
  for (int c1 = 3; c1 < 2 * N - 2; c1 += 2) {
    for (int c3 = 1; c3 <= M; c3 += 1) {
      S1(c0, (c1 + 1) / 2, c3);
      S2(c0, (c1 + 1) / 2, c3);
    }
    for (int c3 = 2; c3 < M; c3 += 1)
      S3(c0, (c1 + 1) / 2, c3);
  }
