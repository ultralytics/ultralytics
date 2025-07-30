for (int c0 = -n + 1; c0 <= n; c0 += 1) {
  if (c0 <= 0)
    for (int c2 = -c0 + 4; c2 <= 2 * n - c0 + 2; c2 += 2)
      S1(1, -c0 + 1, ((c0 + c2) / 2) - 1);
  for (int c1 = max(c0 + 2, -c0 + 4); c1 <= min(2 * n - c0, 2 * n + c0); c1 += 2) {
    for (int c2 = c1 + 2; c2 <= 2 * n + c1; c2 += 2)
      S1((c0 + c1) / 2, (-c0 + c1) / 2, (-c1 + c2) / 2);
    for (int c2 = 1; c2 <= n; c2 += 1)
      S2(((c0 + c1) / 2) - 1, (-c0 + c1) / 2, c2);
  }
  if (c0 >= 1)
    for (int c2 = 1; c2 <= n; c2 += 1)
      S2(n, n - c0 + 1, c2);
}
