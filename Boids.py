import pygame
import random as rand 
import numpy as np 
import heapq 

def norm(array): 
    return array/line_dist(array)

def line_dist(array): 
    return np.sqrt(array[0]**2 + array[1]**2) 

def heuristic(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def astar(grid, start, target):
    width, height = grid.shape
    obstacles = set(zip(*np.where(grid == 1)))
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, target)}
    closed_set = set()
    while open_set:
        current = heapq.heappop(open_set)[1]
        closed_set.add(current)
        neighborhood = [(current[0] - 1, current[1]),(current[0] + 1, current[1]),(current[0], current[1] - 1),(current[0], current[1] + 1),
                         (current[0] - 1, current[1] - 1),(current[0] + 1, current[1] - 1),(current[0] - 1, current[1] + 1),(current[0] + 1, current[1] + 1)]
        if current == target:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        for neighbor in neighborhood:
            if (0 <= neighbor[0] < width and 0 <= neighbor[1] < height and neighbor not in obstacles and neighbor not in closed_set):
                tentative_g_score = g_score[current] + 1
                if (neighbor not in g_score or tentative_g_score < g_score[neighbor]):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = (tentative_g_score + heuristic(neighbor, target))
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

# Boid class
class Boid:
    def __init__(self, x, y, speed, perception, personal_space, use_alignment, use_cohesion, use_separation, use_collision, use_target, use_blocks, use_circles, foci = []):
        self.position = np.array([x,y],dtype = np.float16)
        self.new_position = np.array([0.0,0.0],dtype = np.float16)
        temp = np.array([rand.randint(-100,100), rand.randint(-100,100)],dtype = np.float16)
        self.direction = norm(temp) 
        self.new_direction = np.array([0.0,0.0],dtype = np.float16)
        self.speed = speed
        self.perception = perception
        self.personal_space = personal_space
        self.use_alignment = use_alignment 
        self.use_cohesion = use_cohesion 
        self.use_separation = use_separation 
        self.use_collision = use_collision
        self.use_target = use_target
        if self.use_target == -1: 
            self.color = (0,255,0)
        elif self.use_target == 1: 
            self.color = (255,0,0)
        else: 
            self.color = (0,0,0)
        self.use_blocks = use_blocks 
        self.use_circles = use_circles
        self.foci = foci
        self.path = np.array([])

    def find_boids_sight(self,flock): 
        found = [] 
        for thing in flock: 
            if line_dist(thing.position - self.position) <= self.perception: 
                found.append(thing) 
        return found 
    
    def find_boids_space(self,flock): 
        found = [] 
        for thing in flock: 
            if line_dist(thing.position - self.position) <= self.personal_space: 
                found.append(thing) 
        return found 
    
    def find_closest_boid(self, flock): 
        closest = flock[0] 
        for thing in flock: 
            if line_dist(self.position - thing.position) < line_dist(self.position - closest.position) and thing != self: 
                closest = thing
        return closest

    def update(self):
        self.position = self.new_position
        self.direction = self.new_direction 

    def move(self, flock, blocks, circles, target_x, target_y, grid):
        if self.use_alignment == True or self.use_cohesion == True:
            found_sight = self.find_boids_sight(flock)
        if self.use_separation == True: 
            found_space = self.find_boids_space(flock)
        total_direction = np.array([0.0,0.0],dtype = np.float16)

        if self.use_alignment == True: 
            total_direction += self.align(found_sight)
        if self.use_cohesion == True: 
            total_direction += self.cohere(found_sight)
        if self.use_separation == True: 
            total_direction += self.separate(found_space)
        if self.use_collision == True: 
            total_direction += self.avoid_edge() 
        if self.use_target != 0:
            total_direction += self.target_focus(target_x,target_y) * self.use_target
        if self.use_blocks == True: 
            total_direction +=  self.avoid_block(blocks)
        if self.use_circles == True: 
            total_direction += self.avoid_circle(circles) 
        if len(self.foci) > 0: 
            total_direction += self.foci_vector(grid)

        self.new_direction = norm(self.direction + total_direction)
        while np.isnan(self.new_direction[0]) or np.isnan(self.new_direction[1]): 
            self.new_direction = norm(np.array([rand.randint(-100,100), rand.randint(-100,100)]))
        self.new_position = self.position + (self.new_direction * self.speed)

    def align(self, found_sight):
        flock_direction = np.array([0.0,0.0],dtype = np.float16)
        if len(found_sight) == 1: 
            return flock_direction
        else: 
            for thing in found_sight: 
                    if thing != self:
                        flock_direction += thing.direction 
            return norm(flock_direction) #* (1-line_dist(thing.position - self.position))/self.perception

    def cohere(self, found_sight):
        flock_position = np.array([0.0,0.0],dtype = np.float16)
        if len(found_sight) == 1: 
            return flock_position
        else: 
            for thing in found_sight: 
                flock_position += thing.position 
            flock_position = flock_position/len(found_sight)
            return norm(flock_position - self.position)

    def separate(self, found_space):
        away = self.direction
        close = self.find_closest_boid(found_space)
        if close != self: 
            away += norm(self.position - close.position)
            return norm(away) 
        else: 
            return np.array([0.0,0.0],dtype = np.float16)

    def boundaries(self, blocks, circles):
        width, height = pygame.display.get_surface().get_size()

        if self.new_position[0] < 0:
            self.new_position[0] = width + self.new_position[0]
        elif self.new_position[0] > width:
            self.new_position[0] = self.new_position[0] - width

        if self.new_position[1] < 0:
            self.new_position[1] = height + self.new_position[1]
        elif self.new_position[1] > height:
            self.new_position[1] = self.new_position[1] - height

        if self.use_blocks == True: 
            for block in blocks: 
                if self.new_position[0] > block.params[0] and self.new_position[0] < block.params[0]+block.params[2] and self.new_position[1] > block.params[1] and self.new_position[1] < block.params[1]+block.params[3]:
                    sides = np.array([block.params[0], block.params[0]+block.params[2], block.params[1], block.params[1]+block.params[3]])
                    point = np.array([self.position[0], self.position[0], self.position[1], self.position[1]]) 
                    dists = np.absolute(sides - point)
                    min_dist = np.amin(dists) 
                    index = np.where(dists == min_dist)[0][0] 
                    if index == 0: 
                        self.new_position[0] = block.params[0]
                    elif index == 1: 
                        self.new_position[0] = block.params[0]+block.params[2]
                    elif index == 2: 
                        self.new_position[1] = block.params[1]
                    elif index == 4: 
                        self.new_position[1] = block.params[1]+block.params[3]

        if self.use_circles == True: 
            for circle in circles: 
                if line_dist(circle.position - self.new_position) < circle.radius: 
                    ray = norm(self.new_position - circle.position) 
                    self.new_position = circle.position + ray*(circle.radius + 1)

        if len(self.foci) > 0: 
            if line_dist(self.foci[0] - self.new_position) < 5: 
                self.foci.pop(0) 

    def avoid_edge(self): 
        width, height = pygame.display.get_surface().get_size()
        ray_direction = 1*self.direction 
        ray_position = 1*self.position
        i = 0 
        while ray_position[0] > 0 and ray_position[0] < width and ray_position[1] > 0 and ray_position[1] < height and i < self.personal_space: 
            ray_position += ray_direction 
            i += 1
        if ray_position[0] < 0 or ray_position[0] > width:
            ray_direction[0] = -1*ray_direction[0]
            if ray_direction.all() == -1*self.direction.all(): 
                ray_direction[1] += rand.triangular(-1,1)
        elif ray_position[1] < 0 or ray_position[1] > height:
            ray_direction[1] = -1*ray_direction[1]
            if ray_direction.all() == -1*self.direction.all(): 
                ray_direction[0] += rand.triangular(-1,1)
        else: 
            ray_direction = np.array([0.0,0.0],dtype = np.float16)
        return ray_direction 
    
    def target_focus(self, target_x, target_y): 
        target_position = np.array([target_x, target_y]) 
        if line_dist(target_position - self.position) < self.perception: 
            target_direction = norm(target_position - self.position) 
            return target_direction 
        else: 
            return np.array([0.0,0.0],dtype = np.float16)
        
    def avoid_block(self, blocks): 
        ray_direction = 1*self.direction 
        ray_position = 1*self.position
        direction = 1*self.direction
        i = 0 
        collide = False 
        while i < self.personal_space and collide == False: 
            ray_position += ray_direction 
            for block in blocks: 
                if ray_position[0] > block.params[0] and ray_position[0] < block.params[0]+block.params[2] and ray_position[1] > block.params[1] and ray_position[1] < block.params[1]+block.params[3]:
                    collide = True 
                    collided = block 
            i += 1
        if collide == True:
            sides = np.array([collided.params[0], collided.params[0]+collided.params[2], collided.params[1], collided.params[1]+collided.params[3]])
            point = np.array([ray_position[0], ray_position[0], ray_position[1], ray_position[1]]) 
            dists = np.absolute(sides - point)
            min_dist = np.amin(dists) 
            index = np.where(dists == min_dist)[0][0] 
            if index == 0: 
                direction[0] = -1*direction[0]
            elif index == 1: 
                direction[0] = -1*direction[0]
            elif index == 2: 
                direction[1] = -1*direction[1]
            elif index == 4: 
                direction[1] = -1*direction[1]
            return direction * (3-2.5*(i/self.personal_space))
        else: 
            return np.array([0.0,0.0],dtype = np.float16)

    def avoid_circle(self, circles): 
        ray_direction = 1*self.direction 
        ray_position = 1*self.position
        i = 0 
        collide = False
        while i < self.personal_space and collide == False: 
            ray_position += ray_direction 
            for circle in circles: 
                if line_dist(circle.position - ray_position) < circle.radius: 
                    collide = True 
                    collided = circle
            i += 1
        if collide == True: 
            direction = norm(ray_position - collided.position)
            return direction * (3-2.5*(i/self.personal_space))
        else: 
            return np.array([0.0,0.0],dtype = np.float16)
        
    def foci_vector(self, grid): 
        if line_dist(self.foci[0] - self.position) < 1.5*reduce_factor: 
            return norm(self.foci[0] - self.position)
        else: 
            path = astar(grid, (int(round(self.position[0]/reduce_factor)), int(round(self.position[1]/reduce_factor))), (int(round(self.foci[0][0]/reduce_factor)), int(round(self.foci[0][1]/reduce_factor))))
            if path == None: 
                return np.array([0.0,0.0],dtype = np.float16)
            else: 
                self.path = np.array(path) 
                new_direction = norm(np.array([path[1][0], path[1][1]]) - np.array([int(round(self.position[0]/reduce_factor)), int(round(self.position[1]/reduce_factor))]))
                return new_direction 
    
class spawner: 
    def __init__(self, x, y, spawn_chance):
        self.position = np.array([x,y])
        self.spawn_chance = spawn_chance 
    def spawn(self,flock): 
        var = rand.random() 
        if var <= spawn_chance: 
            flock.append(Boid(self.position[0]+rand.triangular(-4,4), 
                              self.position[1]+rand.triangular(-4,4), 
                              boid_speed, 
                              boid_perception, 
                              boid_personal_space, 
                              use_alignment, 
                              use_cohesion, 
                              use_separation, 
                              use_collision, 
                              use_target, 
                              block_use, 
                              circle_use))

class despawner: 
    def __init__(self, x, y, range):
        self.position = np.array([x,y])
        self.range = range 
    def despawn(self,flock): 
        for boid in flock: 
            if line_dist(self.position - boid.position) <= self.range: 
                flock.remove(boid)

class block: 
    def __init__(self, x, y, width, height): 
        self.params = [x,y,width,height]

class circle: 
    def __init__(self, x, y, radius): 
        self.position = np.array([x,y])
        self.radius = radius 
 
# Initialize pygame
pygame.init()
screen_width, screen_height = 1000, 700
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# Target Vars
target_x = screen_width/2
target_y = screen_height/2
target_dx = 0 
target_dy = 0 
target_speed = 7

# Boids settings
num_boids = 85
boid_speed = 3
boid_perception = 20 
boid_personal_space = 10
flock = []
use_alignment = True 
use_cohesion = True
use_separation = True
use_collision = False
use_target = 0
foci =  [np.array([int(screen_width/6),int(screen_height/6)]), 
        np.array([int(5*screen_width/6),int(screen_height/6)]),
        np.array([int(screen_width/6),int(5*screen_height/6)]),
        np.array([int(5*screen_width/6),int(5*screen_height/6)])]
render_boids = True
render_target = False
render_foci = True
render_path = True

# Spawner settings
spawn_use = False
spawn_chance = .05 
spawn_x = screen_width/6
spawn_y = screen_height/4
spawners = [] 
render_spawners = False

# Dewpawner settings
despawn_use = False 
despawn_x = 5*screen_width/6
despawn_y = 3*screen_height/4
despawners = [] 
despawn_range = 40
render_despawners = False

# Block settings 
block_use = True 
blocks = [] 
render_blocks = False

# Circle settings 
circle_use = False
circles = [] 
render_circles = False

# Pathfind settings  
grid = np.zeros((screen_width,screen_height))
reduce_factor = 10
render_grid = True

# Create boids
for i in range(num_boids):
    flock.append(Boid(rand.randint(int(screen_width/3),int(2*screen_width/3))+rand.random(),       # float, x position
                      rand.randint(int(screen_height/3),int(2*screen_height/3))+rand.random(),      # float, y position
                      boid_speed,                                       # int, speed 
                      boid_perception,                                  # int, perception 
                      boid_personal_space,                              # int, personal space
                      use_alignment,                                    # bool, True to use the alignment vector
                      use_cohesion,                                     # bool, True to use the cohesion vector
                      use_separation,                                   # bool, True to use the separation vector 
                      use_collision,                                    # bool, True to have the boids avoid the edges of the map 
                      use_target,                                       # int, 1 to move toward the target, 0 to ignore the target, -1 to move away from target
                      block_use,                                        # bool, True to avoid blocks
                      circle_use,                                       # bool, True to avoid circles
                      list(foci)))                                      # list, a list of destitations that the boid needs to travel to 

# Create spawner 
if spawn_use == True: 
    spawners.append(spawner(spawn_x, spawn_y, spawn_chance))

# Create despawner 
if despawn_use == True: 
    despawners.append(despawner(despawn_x, despawn_y, despawn_range))

# Create Blocks 
if block_use == True: 
    blocks.append(block((screen_width/3)-5, 0, 50, screen_height/3))
    blocks.append(block((2*screen_width/3)-5, 2*screen_height/3, 50, screen_height/3))
    blocks.append(block((screen_width/3)-5, 2*screen_height/3, 50, screen_height/3))
    blocks.append(block((2*screen_width/3)-5, 0, 50, screen_height/3))
    for i in blocks: 
        x1 = int(i.params[0])
        x2 = int(i.params[0]+i.params[2]+1)
        y1 = int(i.params[1])
        y2 = int(i.params[1]+i.params[3]+1)
        grid[x1:x2, y1:y2] = 1

# Create Circles
if circle_use == True: 
    for i in range(7): 
        circles.append(circle(rand.randint(0,screen_width), rand.randint(0,screen_height), rand.randint(20,60)))              

# Grid Reduction 
if reduce_factor != 0: 
    reduced_grid = np.zeros((int(round(screen_width/reduce_factor)),int(round(screen_height/reduce_factor)))) 
    for x in range(int(screen_width/reduce_factor)): 
        for y in range(int(screen_height/reduce_factor)): 
            value = np.sum(grid[x*reduce_factor:(x+1)*reduce_factor,y*reduce_factor:(y+1)*reduce_factor])
            if value > (reduce_factor**2)/2: 
                reduced_grid[x][y] = 1
else: 
    reduced_grid = grid

# Main loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN: 
            if event.key == pygame.K_ESCAPE: 
                running = False
    if use_target != 0: 
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT: 
                target_dx = -1*target_speed
            if event.key == pygame.K_RIGHT: 
                target_dx = target_speed
            if event.key == pygame.K_UP:
                target_dy = -1*target_speed
            if event.key == pygame.K_DOWN: 
                target_dy = target_speed
        if event.type == pygame.KEYUP: 
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT: 
                target_dx = 0 
            if event.key == pygame.K_UP or event.key == pygame.K_DOWN: 
                target_dy = 0 
        if target_x + target_dx > screen_width :
            target_x = 0 
        elif target_x + target_dx < 0 :
            target_x = screen_width 
        else:
            target_x = target_x + target_dx
        if target_y + target_dy > screen_height :
            target_y = 0 
        elif target_y + target_dy < 0 :
            target_y = screen_height
        else:
            target_y = target_y + target_dy

    # Update boids
    for boid in flock: 
        boid.move(flock, blocks, circles, target_x, target_y, reduced_grid)
        boid.boundaries(blocks, circles) 
        boid.update()

    # Update Spawners 
    if spawn_use == True: 
        for spawner in spawners: 
            spawner.spawn(flock)

    # Update Despawners 
    if despawn_use == True: 
        for despawner in despawners: 
            despawner.despawn(flock) 

    # Screen Vars
    screen.fill((255, 255, 255))

    # Render Grid 
    if render_grid == True: 
        bad = set(zip(*np.where(grid == 1))) 
        for i in bad: 
            pygame.draw.circle(screen, (200,200,200), i, 1)

    # Render Target
    if use_target != 0 and render_target == True:
        pygame.draw.circle(screen,(255,0,0),(target_x, target_y), 5)

    # Render Spawners 
    if spawn_use == True and render_spawners == True: 
        for spawner in spawners: 
            pygame.draw.circle(screen, (0,0,255), (spawner.position[0], spawner.position[1]), 10) 

    # Render Despawners 
    if despawn_use == True and render_despawners == True: 
        for despawner in despawners: 
            pygame.draw.circle(screen, (255,0,0), (despawner.position[0], despawner.position[1]), 10)

    # Render Blocks 
    if block_use == True and render_blocks == True: 
        for block in blocks: 
            pygame.draw.rect(screen, (200, 100, 0), block.params)

    # Render Circles 
    if circle_use == True and render_circles == True: 
        for circle in circles: 
            pygame.draw.circle(screen, (0,255,150), circle.position, circle.radius)

    # Render Foci 
    if len(foci) > 0 and render_foci == True: 
        for i in foci: 
            pygame.draw.circle(screen, (200,0,250), i, 5)

    # Render Paths 
    if render_path == True: 
        for boid in flock: 
            for i in boid.path: 
                pygame.draw.circle(screen, (150,10,150), i*reduce_factor, 1)

    # Render boids
    if render_boids == True: 
        for boid in flock:
            pygame.draw.line(screen, boid.color, 
                            (int(boid.position[0])-int(5*boid.direction[0]), 
                            int(boid.position[1])-(int(5*boid.direction[1]))), 
                            (int(boid.position[0])+int(5*boid.direction[0]), 
                            int(boid.position[1])+(int(5*boid.direction[1]))), 
                            3)

    pygame.display.flip()
    clock.tick(60)

# Quit the program
pygame.quit()
