{
  S1(0, 0);
  for (int c0 = 1; c0 <= N; c0 += 1) {
    S2(c0, 0);
    for (int c1 = 1; c1 < c0; c1 += 1)
      S6(c0, c1);
    S3(c0, c0);
  }
  S7(N + 1, 0);
  for (int c1 = 1; c1 <= N; c1 += 1) {
    S6(N + 1, c1);
    S8(N + 1, c1);
  }
  for (int c0 = N + 2; c0 < 2 * M - N - 1; c0 += 1) {
    S7(c0, -N + (N + c0 + 1) / 2 - 1);
    if ((N - c0) % 2 == 0) {
      S5(c0, (-N + c0) / 2);
      S8(c0, (-N + c0) / 2);
    }
    for (int c1 = -N + (N + c0) / 2 + 1; c1 < (N + c0 + 1) / 2; c1 += 1) {
      S6(c0, c1);
      S8(c0, c1);
    }
    if ((N - c0) % 2 == 0) {
      S4(c0, (N + c0) / 2);
      S8(c0, (N + c0) / 2);
    }
  }
  for (int c0 = 2 * M - N - 1; c0 < 2 * M - 1; c0 += 1)
    for (int c1 = -M + c0 + 1; c1 < M; c1 += 1)
      S6(c0, c1);
}
