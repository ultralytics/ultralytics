{
  for (int c0 = 1; c0 <= N; c0 += 1)
    S1(c0);
  for (int c0 = N + 1; c0 <= 2 * N; c0 += 1)
    for (int c1 = 1; c1 <= N; c1 += 1)
      S2(c1, -N + c0);
  for (int c0 = 2 * N + 1; c0 <= M + N; c0 += 1) {
    for (int c1 = 1; c1 <= N; c1 += 1)
      S3(c1, -2 * N + c0);
    for (int c1 = 1; c1 <= N; c1 += 1)
      S2(c1, -N + c0);
  }
  for (int c0 = M + N + 1; c0 <= M + 2 * N; c0 += 1)
    for (int c1 = 1; c1 <= N; c1 += 1)
      S3(c1, -2 * N + c0);
}
