{
  for (int c0 = M + 1; c0 <= 2 * M; c0 += 1)
    S1(-M + c0);
  for (int c0 = 2 * M + 1; c0 <= M + N; c0 += 1) {
    S2(-2 * M + c0);
    S1(-M + c0);
  }
  for (int c0 = M + N + 1; c0 <= 2 * M + N; c0 += 1)
    S2(-2 * M + c0);
}
