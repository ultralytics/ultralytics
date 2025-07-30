{
  for (int c0 = 1; c0 <= n; c0 += 1) {
    S1(c0, c0);
    for (int c1 = c0; c1 <= n; c1 += 1)
      S2(c0, c1);
    S3(c0, n);
  }
  for (int c0 = n + 1; c0 <= m; c0 += 1)
    S3(c0, n);
}
