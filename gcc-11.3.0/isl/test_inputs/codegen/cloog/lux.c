for (int c0 = 1; c0 <= M; c0 += 1) {
  for (int c1 = 1; c1 < c0; c1 += 1)
    for (int c2 = c1 + 1; c2 <= M; c2 += 1)
      S2(c0, c1, c2, c2, c0);
  for (int c3 = c0 + 1; c3 <= M; c3 += 1)
    S1(c0, c0, M, c3);
}
