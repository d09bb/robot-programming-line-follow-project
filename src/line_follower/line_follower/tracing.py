"""Directed traversal of thin observed lines, retaining visible loops."""
import math
import numpy as np


def trace(skeleton, start, initial=None, prior=None):
    cells = set(map(tuple,np.argwhere(skeleton)))
    def neighbors(cell):
        r,c = cell
        result = []
        for dr,dc in ((1,0),(1,1),(1,-1),(0,1),(0,-1),(-1,0),(-1,1),(-1,-1)):
            nxt = r+dr,c+dc
            if nxt not in cells:
                continue
            # Remove triangular diagonal shortcuts around a one-cell corner.
            if dr and dc and ((r+dr,c) in cells or (r,c+dc) in cells):
                continue
            result.append(nxt)
        return result
    adjacent = {cell:neighbors(cell) for cell in cells}
    vertices = {cell for cell,links in adjacent.items() if len(links)!=2}
    vertices.add(start)
    links = {cell:[] for cell in vertices}
    used = set()
    edges = []
    for vertex in sorted(vertices):
        for nxt in adjacent[vertex]:
            key = frozenset((vertex,nxt))
            if key in used:
                continue
            route = [vertex,nxt]
            used.add(key)
            while nxt not in vertices:
                next_cell = next(cell for cell in adjacent[nxt] if cell!=route[-2])
                used.add(frozenset((nxt,next_cell)))
                route.append(next_cell)
                nxt = next_cell
            index = len(edges)
            points = np.asarray(route,dtype=float)
            length = float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())*.005
            edges.append((route,length))
            links[vertex].append((index,False,nxt))
            links[nxt].append((index,True,vertex))
    endpoints = {cell for cell in vertices if len(adjacent[cell])==1 and cell!=start}
    initial = np.array([1.,0.]) if initial is None else np.asarray(initial)
    initial = initial/max(np.linalg.norm(initial),1e-9)
    candidates = []
    # Edge trails allow an observed circular arm to return to a junction once;
    # a shortest-path tree discards precisely that part of a horseshoe.
    stack = [(start,[],frozenset(),initial,0.,0.)]
    explored = 0
    while stack and explored < 12000:
        vertex,path,taken,heading,length,bending = stack.pop()
        explored += 1
        if path and (vertex in endpoints or (not endpoints and vertex==start)):
            coverage = 0.
            if prior is not None and len(prior)>=2:
                sampled = np.asarray(([start]+path)[::4],dtype=float)
                distances = np.linalg.norm(sampled[:,None,:]-prior[None,:,:],axis=2)
                matches = np.argmin(distances,axis=1)
                close = distances[np.arange(len(sampled)),matches] <= 8.
                arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(prior,axis=0),axis=1))]*.005
                progress = arc[matches[close]]
                if len(progress)>2:
                    backward = float(np.max(np.maximum.accumulate(progress)-progress))
                    if backward > .08:
                        continue
                    coverage = float(np.max(progress)-np.min(progress))
            candidates.append((coverage,vertex[0]*.005,length-.015*bending,path))
            continue
        if length>3.:
            continue
        for index,reverse,end in links.get(vertex,[]):
            if index in taken:
                continue
            route,edge_length = edges[index]
            route = route[::-1] if reverse else route
            first = np.array(route[min(6,len(route)-1)],float)-route[0]
            first /= max(np.linalg.norm(first),1e-9)
            angle = math.acos(float(np.clip(first@heading,-1.,1.)))
            if angle>math.radians(125):
                continue
            last = np.array(route[-1],float)-route[max(0,len(route)-7)]
            last /= max(np.linalg.norm(last),1e-9)
            stack.append((end,path+route[1:],taken|{index},last,
                          length+edge_length,bending+angle*angle))
    if not candidates:
        return []
    # A one-cell endpoint change must not outweigh an entire observed arc.
    # Prefer consistent progress on the previous route, with a 3 cm deadband.
    coverage = max(item[0] for item in candidates)
    candidates = [item for item in candidates if item[0]>=coverage-.03]
    forward = max(item[1] for item in candidates)
    candidates = [item for item in candidates if item[1]>=forward-.03]
    best = max(candidates,key=lambda item:(item[2],tuple(item[3])))
    return [start]+best[3]


def join_fragments(skeleton):
    """Bridge at most 4 cm between facing endpoints, never parallel arms."""
    import cv2
    offsets = ((1,0),(1,1),(1,-1),(0,1),(0,-1),(-1,0),(-1,1),(-1,-1))
    cells = set(map(tuple,np.argwhere(skeleton)))
    def neighbors(cell):
        return [(cell[0]+dr,cell[1]+dc) for dr,dc in offsets
                if (cell[0]+dr,cell[1]+dc) in cells]
    _,labels = cv2.connectedComponents(skeleton,8)
    endpoints = []
    for cell in sorted(cells):
        adjacent = neighbors(cell)
        if len(adjacent)!=1:
            continue
        route = [cell,adjacent[0]]
        for _ in range(5):
            options = [n for n in neighbors(route[-1]) if n!=route[-2]]
            if len(options)!=1:
                break
            route.append(options[0])
        tangent = np.array(cell,dtype=float)-route[-1]
        length = np.linalg.norm(tangent)
        if length>=2.:
            endpoints.append((cell,tangent/length))
    pairs = []
    for i,(a,ta) in enumerate(endpoints):
        for j,(b,tb) in enumerate(endpoints[i+1:],i+1):
            delta = np.array(b,dtype=float)-a
            distance = np.linalg.norm(delta)
            if labels[a]==labels[b] or not 1. < distance <= 8.:
                continue
            direction = delta/distance
            if direction@ta>=.8 and -direction@tb>=.8:
                pairs.append((distance,i,j))
    linked = set()
    result = skeleton.copy()
    for _,i,j in sorted(pairs):
        if i in linked or j in linked:
            continue
        a,b = endpoints[i][0],endpoints[j][0]
        cv2.line(result,(a[1],a[0]),(b[1],b[0]),1,1)
        linked.update((i,j))
    return result
