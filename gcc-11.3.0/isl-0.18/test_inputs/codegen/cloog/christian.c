for (int c0 = -N + 1; c0 <= N; c0 += 1) {
  for (int c1 = max(0, c0); c1 < min(N, N + c0); c1 += 1)
    S1(c1, -c0 + c1);
  for (int c1 = max(0, c0 - 1); c1 < min(N, N + c0 - 1); c1 += 1)
    S2(c1, -c0 + c1 + 1);
}
