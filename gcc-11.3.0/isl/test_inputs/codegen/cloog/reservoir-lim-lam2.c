{
  for (int c0 = 1; c0 <= M; c0 += 1)
    S1(c0);
  for (int c0 = 1; c0 <= M; c0 += 1)
    for (int c1 = 2; c1 <= N; c1 += 1)
      S2(c0, c1);
  for (int c0 = 1; c0 <= M; c0 += 1)
    for (int c1 = 1; c1 < N; c1 += 1)
      S3(c0, c1);
}
