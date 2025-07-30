for (int c0 = 1; c0 < 2 * M - 1; c0 += 1) {
  for (int c1 = max(-M + 1, -c0 + 1); c1 < 0; c1 += 1) {
    for (int c3 = max(1, -M + c0 + 1); c3 <= min(M - 1, c0 + c1); c3 += 1)
      S1(c3, c0 + c1 - c3, -c1);
    for (int c2 = max(-M + c0 + 1, -c1); c2 < min(M, c0); c2 += 1)
      S2(c0 - c2, c1 + c2, c2);
  }
  for (int c3 = max(1, -M + c0 + 1); c3 <= min(M - 1, c0); c3 += 1)
    S1(c3, c0 - c3, 0);
}
