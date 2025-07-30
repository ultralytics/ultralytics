for (int c0 = 0; c0 <= 199; c0 += 1) {
  for (int c1 = 50 * c0; c1 <= 50 * c0 + 24; c1 += 1)
    for (int c2 = 0; c2 <= c1; c2 += 1)
      S1(c0, c1, c2);
  for (int c1 = 50 * c0 + 25; c1 <= 50 * c0 + 49; c1 += 1)
    for (int c2 = 0; c2 <= c1; c2 += 1)
      S2(c0, c1, c2);
}
