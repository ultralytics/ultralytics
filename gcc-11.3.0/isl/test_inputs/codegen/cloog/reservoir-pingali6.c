for (int c0 = 1; c0 <= M; c0 += 1) {
  for (int c2 = 2; c2 < N; c2 += 1)
    for (int c3 = 2; c3 < N; c3 += 1)
      S1(c0, c2, c3);
  for (int c2 = 2; c2 < N; c2 += 1)
    for (int c3 = 2; c3 < N; c3 += 1)
      S2(c0, c2, c3);
}
